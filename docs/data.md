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

Run

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
| `fineweb` | `--target-source-tokens` selects a shuffled document prefix |
| `finemath` | `--subset finemath-3plus`, `finemath-4plus`, `infiwebmath-3plus`, `infiwebmath-4plus` |
| `dolmino` | `--subset`; run `dolmino --help` for available subsets |
| `openmathinstruct2`, `openmathreasoning`, `natural_reasoning`, `stackexchange` | One workflow per source |
| `opencodereasoning` | `--subset split_0` or `split_1` |
| `stackedu` | `--subset` selects the language to tokenize and pack |
| `jupyter` | Rehydrates notebooks and converts them into script text |
| `issues_kaggle` | `--subset issues` or `kaggle` |
