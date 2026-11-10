# Data preparation

## Controlled setup

### Tokenizer
For controlled experiments (140M/370M), we train our own tokenizers on the
FineWeb-Edu dataset, following Parcae. 

To train a tokenizer from raw Parquet shards:

```bash
python scripts/data/train_parcae_tokenizer.py --data-dir /path/to/raw/parquet
```

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

The default recipe, `configs/data/mix.yaml`, expands into 34 training-source datasets, including web documents, math and coding domains,
and prepares the validation data from Fineweb-edu. The processed data is stored in `data/packed/<source>/`.

Override shared settings or choose a custom preparation recipe:

```bash
python scripts/data/create_mix_data.py all --recipe configs/data/mix.yaml --workers 16
```

Single-source data preprocessing is also available:

```bash
python scripts/data/create_mix_data.py openmathinstruct2 --stage all
```

Check all source options without downloading or writing data:

```bash
python scripts/data/create_mix_data.py all --dry-run
```

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
