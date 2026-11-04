"""Shared physical Parquet writer for globally shuffled HF datasets."""

import os
import json
import time
from pathlib import Path
from typing import Any
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

_SHUFFLED_DATASET: Dataset | None = None


def expected_shard_rows(total_rows: int, num_shards: int, shard_id: int) -> int:
    rows_per_shard, remainder = divmod(total_rows, num_shards)
    return rows_per_shard + int(shard_id < remainder)


def inspect_parquet(path: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(path)
    return {
        "path": str(path),
        "rows": parquet_file.metadata.num_rows,
        "row_groups": parquet_file.num_row_groups,
        "size_bytes": path.stat().st_size,
        "schema_names": parquet_file.schema_arrow.names,
    }


def write_one_shard(task: tuple[int, int, str, int, str, int]) -> dict[str, Any]:
    shard_id, num_shards, output_text, batch_rows, compression, compression_level = task
    if _SHUFFLED_DATASET is None:
        raise RuntimeError("Worker did not inherit the shuffled dataset")

    pa.set_cpu_count(1)
    output_path = Path(output_text)
    shard = _SHUFFLED_DATASET.shard(
        num_shards=num_shards,
        index=shard_id,
        contiguous=True,
    )
    expected_rows = len(shard)
    if output_path.is_file():
        existing = inspect_parquet(output_path)
        if (
            existing["rows"] == expected_rows
            and existing["schema_names"] == _SHUFFLED_DATASET.column_names
        ):
            existing.update({"shard_id": shard_id, "reused": True})
            return existing
        raise ValueError(f"Refusing to reuse invalid shard: {output_path}")

    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    started = time.time()
    arrow_shard = shard.with_format("arrow")
    with pq.ParquetWriter(
        temporary,
        schema=shard.features.arrow_schema,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=True,
        write_statistics=True,
    ) as writer:
        for batch in arrow_shard.iter(batch_size=batch_rows):
            writer.write_table(batch, row_group_size=batch_rows)
    os.replace(temporary, output_path)

    result = inspect_parquet(output_path)
    if result["rows"] != expected_rows:
        raise ValueError(
            f"Written shard {output_path} has {result['rows']} rows, "
            f"expected {expected_rows}"
        )
    if result["schema_names"] != _SHUFFLED_DATASET.column_names:
        raise ValueError(f"Written shard {output_path} has an unexpected schema")
    result.update(
        {
            "elapsed_seconds": time.time() - started,
            "reused": False,
            "shard_id": shard_id,
        }
    )
    return result


def row_value(row: dict) -> str:
    """Compare serialized values directly, including nested fields and NaNs."""
    return json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
