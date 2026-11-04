import re
import shutil
from pathlib import Path
from typing import Optional

from recpre.path_utils import resolve_within_root


def _checkpoint_step(path: Path, run_name: str) -> Optional[int]:
    """Return the logical checkpoint step when path belongs to run_name."""
    match = re.fullmatch(r"step-(\d+)-(.+)", path.name)
    if match is None:
        return None

    checkpoint_tail = match.group(2)
    valid_tails = {run_name, f"{run_name}.pt", f"{run_name}.pth"}
    if checkpoint_tail not in valid_tails and re.fullmatch(
        rf"{re.escape(run_name)}_\d+\.pth", checkpoint_tail
    ) is None:
        return None
    return int(match.group(1))


def prune_old_checkpoints(
    checkpoint_dir: str | Path,
    run_name: str,
    max_checkpoints_to_keep: Optional[int],
    *,
    allowed_root: str | Path,
) -> list[Path]:
    """Delete all artifacts belonging to checkpoint steps older than the newest N."""
    if max_checkpoints_to_keep is None:
        return []
    if (
        not isinstance(max_checkpoints_to_keep, int)
        or isinstance(max_checkpoints_to_keep, bool)
        or max_checkpoints_to_keep <= 0
    ):
        raise ValueError("max_checkpoints_to_keep must be a positive integer or None")

    checkpoint_dir = resolve_within_root(
        checkpoint_dir,
        allowed_root,
        label="checkpoint pruning directory",
    )
    if not checkpoint_dir.exists():
        return []

    checkpoints_by_step: dict[int, list[Path]] = {}
    for path in checkpoint_dir.rglob("step-*"):
        step = _checkpoint_step(path, run_name)
        if step is not None:
            checkpoints_by_step.setdefault(step, []).append(path)

    stale_steps = sorted(checkpoints_by_step)[:-max_checkpoints_to_keep]
    deleted: list[Path] = []
    for step in stale_steps:
        for path in sorted(checkpoints_by_step[step], key=lambda item: len(item.parts), reverse=True):
            try:
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except FileNotFoundError:
                continue
            deleted.append(path)
    return deleted
