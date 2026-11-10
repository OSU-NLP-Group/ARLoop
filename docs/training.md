# Training

Use the YAML files under `configs/fineweb_edu/` and `configs/mixed_data/`.

## Controlled experiments
`configs/fineweb_edu/` contains training configs for the 140M/370M experiments.
The model architecture uses transformer blocks with multi-head attention and
an MLP as the feed-forward sublayer.
Following Parcae and nanochat, we use value embeddings, with each physical
layer learning distinct embeddings.
The context length is 2048. The vocabulary size is 32768.
We train on FineWeb-Edu with a cosine LR scheduler. The global batch size is 256.

To train a model, run:

```bash
bash scripts/train/run_train.sh configs/fineweb_edu/140m_arloop_full_r2.yaml 4
```

### Online HF datasets

To train on FineWeb-Edu with online tokenization and best-fit packing:

```bash
bash scripts/train/run_train.sh configs/fineweb_edu/140m_arloop_full_r2_online.yaml 4
```

This recipe keeps the original model and optimizer. Training streams the first
92 FineWeb-Edu shards; validation streams the remaining eight, matching the
file-disjoint holdout used by offline preparation. Neither split requires locally
prepared packed files.

For another HF text dataset, configure `data_config.train_data` as follows:

```yaml
train_data:
  - type: stream
    prefix: corpus
    dataset: organization/dataset
    dataset_config: null
    dataset_split: train
    text_column: text
    packing_buffer_size: 1000
    shuffle_buffer_size: 0
```

`data_files` optionally selects source files within the dataset. Parquet sources use
synchronous range reads; other formats use Hugging Face's streaming reader. Text
columns must contain plain strings. Keep `dataloader_num_workers: 0`; streaming
supports one source per train/validation loader and requires at least one shard per
data rank in each split. The FineWeb-Edu online recipe therefore supports up to eight
data ranks.

Online and offline preparation share the same best-fit implementation, including BOS,
length tie-breaking, document cropping, and final-row handling. Identical tokenized
documents in identical order produce identical packed rows. Online shard partitioning
does not reproduce the offline lane and row-group shuffle order.

A positive `shuffle_buffer_size` enables document shuffling. Checkpoints save its
buffer and RNG, the packing buffer, and the source position. Resume requires the same
data-rank count, source files and contents, tokenizer, and packing settings.

Validation accepts either `packed-bin` or `stream` using the same source fields.
A streaming validation loader tokenizes and packs only the prefix needed by
`eval_num_sequences` (or `eval_iters`), then closes the reader. The resulting CPU
batches are reused on every validation; a fresh iterator restarts deterministically.
`cache_validation_batches: false` disables persistent disk caching, not the in-memory
cache used within a training run.

## Realistic pretraining experiments
`configs/mixed_data/` contains training configs for the ARLoop-0.6B and
ARLoop-1.2B experiments.
The model architecture uses transformer blocks with grouped-query attention
and SwiGLU as the feed-forward sublayer.
Following Parcae and nanochat, we use value embeddings shared across physical layers.
The context length is 4096. The vocabulary size is 49152.
We train on a mixture of pretraining sources with a cosine LR scheduler. The global batch size is 256 for ARLoop-0.6B and 512 for ARLoop-1.2B.

To train a model, run:

```bash
bash scripts/train/run_train.sh configs/mixed_data/0p6b_arloop_r3_180b.yaml 8
```

### Data scheduling
Training ARLoop-0.6B and ARLoop-1.2B involves three stages, each with different
sampling proportions for the data sources.
This schedule is motivated by the recipes of [SmolLM](https://huggingface.co/blog/smollm3) and [OLMo](https://arxiv.org/abs/2501.00656).
The figure below provides an overview of our data schedule.

![Three-stage pretraining data mixture and learning-rate schedule](../assets/pretraining_recipe.png)
