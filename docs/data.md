# Data preparation

Training data are not included in this repository. The exact tokenizer files are
included in `tokenizer/parcae_tokenizer/` and `tokenizer/smollm2_tokenizer/`.
Do not retrain or substitute a tokenizer when reproducing an existing checkpoint.

## Controlled setup

### Tokenizer
For controlled experiments (140M/370M), we train our own tokenizers on the
FineWeb-Edu dataset, following Parcae.

To train a tokenizer from raw Parquet shards:

```bash
python scripts/data/train_parcae_tokenizer.py --data-dir /path/to/raw/parquet
```

Defaults: a 32,768-token vocabulary, a 2B-character training budget, and a
10,000-character cap per document. The last sorted shard is held out when more
than one shard is present; use `--holdout-files 0` for an existing training-only
split. Outputs are saved to `outputs/parcae_tokenizer/`, including the Hugging
Face tokenizer files and `token_bytes.pt`. Use the bundled tokenizer when
reproducing the released checkpoints; retraining does not guarantee identical
token IDs.

### Data Preprocessing
We use the shuffled [FineWeb-Edu-100B](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle)
as the data source. After installing the `data` dependencies, run

```bash
python scripts/data/create_fineweb_edu.py
```
to run the complete data preprocessing pipeline. This downloads the corpus,
creates a file-disjoint train/validation split, tokenizes each file, packs
documents into fixed-length sequences (e.g., 2048 tokens), and stores the
resulting data as `.bin` files. Packing uses a best-fit strategy, following the
implementations in nanochat and Parcae.

## Realistic pretraining setup

### Tokenizer
For ARLoop-0.6B and ARLoop-1.2B experiments, we use SmolLM2's tokenizer,
following Ouro's recipe.

### Data Preprocessing
We train ARLoop-0.6B and ARLoop-1.2B on a diverse mixture drawn from the
[SmolLM3 pretraining data sources](https://huggingface.co/collections/HuggingFaceTB/smollm3-pretraining-datasets).
For each source, we provide a workflow to download, tokenize, pack, and store
the resulting data as `.bin` files. To match our large-scale pretraining setup,
packing concatenates documents into fixed-length sequences rather than using
a best-fit strategy.

Prepare every source and subset with one command:

```bash
python scripts/data/create_mix_data.py all
```

The default recipe, `configs/data/mix.yaml`, expands into 34 training-source tasks
and also prepares the FineWeb validation data. Tasks run sequentially; existing
completed stages are reused. A failed task stops the command, and running it again
skips completed stages. This is stage-level continuation, not exact recovery of every
partially completed operation.

Check all source options without downloading or writing data:

```bash
python scripts/data/create_mix_data.py all --dry-run
```

Override shared settings or choose a custom preparation recipe:

```bash
python scripts/data/create_mix_data.py all --data-root /path/to/data --workers 16
python scripts/data/create_mix_data.py all --recipe configs/data/mix.yaml --stage tokenize
```

The recipe records source subsets and source-specific options. Recorded upstream
commits are retained; dependencies without recorded commits default to `main`.
Set their `revision` (and OpenCodeReasoning's `apps_revision` / `taco_revision`)
in the recipe to select fixed upstream versions. Access to the upstream repositories
and rehydration services is still required.

FineWeb uses the full 1.3T source corpus. A 20M source-token prefix of complete
shuffled documents is reserved for validation first; all remaining documents,
including the rest of the validation boundary bucket, are tokenized for training.
There is no 700B training-prefix limit. The held-out prefix is concatenation-packed
to approximately 10M output tokens, rounded down to complete 4097-token records.
All reserved documents are excluded from training, including any tokens unused by
the validation packing budget. This ensures disjoint source rows, not cross-document text deduplication.

Packed outputs use one directory layout:

```text
data/packed/<source>/<optional_subset>/smollm2_seq4096_s<seed>/<train_or_val>/
```

Source names match the CLI; subset directory names use lowercase letters and
underscores (for example, `tinyGSM-MIND` becomes `tinygsm_mind`). CLI subset
arguments retain their upstream spelling. Examples with the default seed:

```text
data/packed/fineweb/smollm2_seq4096_s42/train/
data/packed/fineweb/heldout_10m/smollm2_seq4096_s42/val/
data/packed/dolmino/basic_math/smollm2_seq4096_s42/train/
data/packed/stackedu/python/smollm2_seq4096_s42/train/
```

Raw downloads and tokenized intermediate caches keep their existing paths.
Existing packed directories are not moved automatically; move them to the new
locations or update the training configuration to use their existing paths.

Each source remains separate on disk; mixture weights and stage schedules are applied
by the training dataloader. A custom `--data-root` must also be reflected in the
training configuration's data paths.

Single-source workflows remain available:

```bash
python scripts/data/create_mix_data.py --help
python scripts/data/create_mix_data.py openmathinstruct2 --help
python scripts/data/create_mix_data.py openmathinstruct2 --stage all
python scripts/data/create_mix_data.py dolmino --subset basic_math --revision COMMIT_SHA
python scripts/data/create_mix_data.py finemath --subset finemath-3plus --revision COMMIT_SHA
```
to process each data source.

### Data sources

| Source | Subset selection |
|---|---|
| `fineweb` | Full 1.3T corpus minus a separately reserved validation prefix |
| `finemath` | `--subset finemath-3plus`, `finemath-4plus`, `infiwebmath-3plus`, `infiwebmath-4plus` |
| `dolmino` | `--subset`; run `dolmino --help` for available subsets |
| `openmathinstruct2`, `openmathreasoning`, `natural_reasoning`, `stackexchange` | One workflow per source |
| `opencodereasoning` | `--subset split_0` or `split_1` |
| `stackedu` | `--subset` selects the language to tokenize and pack |
| `jupyter` | Rehydrates notebooks and converts them into script text |
| `issues_kaggle` | `--subset issues` or `kaggle` |
