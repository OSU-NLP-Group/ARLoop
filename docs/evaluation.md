# Evaluation

## Reproducing results of controlled experiments

Use the `arloop_controlled` task group with zero-shot evaluation, a context
length of 2048, and the recurrence setting used during training. The group contains ARC-C, ARC-E, HellaSwag, LAMBADA OpenAI, LAMBADA standard, OBQA, PIQA,
RACE, SciQ and WinoGrande. Average `acc_norm` for HellaSwag/OBQA/PIQA and `acc`
for the other seven tasks.

## Reproducing results of realistic pretraining experiments

| Task | Few-shot | Reported metric |
|---|---:|---|
| ARC-C | 25 | acc_norm |
| HellaSwag | 10 | acc_norm |
| WinoGrande | 5 | acc |
| OpenBookQA | 0 | acc_norm |
| GSM8K CoT | 5 | flexible-extract exact match |
| Minerva MATH | 4 | math_verify |
| MBPP | 0 | pass@1 |
| HumanEval | 0 | pass@1 |

Use BF16, a context length of 4096, greedy generation with at most 1024 new
tokens, and no chat template. The evaluation scripts load original HF or native
training weights directly into the vLLM model. Only temporary configuration JSON
files are created; no additional weight checkpoint is written.

```bash
python scripts/evaluate/evaluate_all_benchmarks_vllm.py --allow-code-execution
# Another HF checkpoint:
python scripts/evaluate/evaluate_all_benchmarks_vllm.py \
  --model osunlp/ARLoop-1.2B --revision step-010000 --allow-code-execution
# A native training checkpoint:
python scripts/evaluate/evaluate_all_benchmarks_vllm.py \
  --checkpoint /path/to/run/checkpoints-DDPStrategy/step-XXXXXXXX --allow-code-execution
```

Both evaluation entrypoints accept these source options. The default model is
`osunlp/ARLoop-1.2B`. Use `--model` for another HF repository or local HF directory.
For native input, pass `--checkpoint`: the training run directory is inferred
from its ancestors, and the tokenizer path is read from `run_config.json`.
The recorded tokenizer path must be accessible on the evaluation machine.
Optionally, `--run-dir` specifies the training run directory explicitly and,
without `--checkpoint`, selects its latest checkpoint.
FSDP shard directories and legacy pickles requiring removed Python modules are
not supported. Use `--generation-only` to save code generations without executing
them.
