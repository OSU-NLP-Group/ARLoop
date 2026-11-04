from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


COMMON_STOPS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
]
DATASET_STOPS = {
    "humaneval": ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert "],
    "mbpp": ['\n"""', "\nassert"],
}


def prepare_evalplus_prompt(prompt: str, dataset_name: str | None = None) -> str:
    """Paper code protocol: MBPP ends in a newline, HumanEval is stripped."""
    return prompt.strip() + ("\n" if dataset_name == "mbpp" else "")


def _generate_vllm(llm: Any, prompts: list[str], max_new_tokens: int,
                   stops: list[str]) -> Iterator[str]:
    from vllm import SamplingParams

    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=max_new_tokens, stop=stops, seed=0,
    )
    for output in llm.generate(prompts, sampling, use_tqdm=True):
        yield output.outputs[0].text


def run_evalplus_dataset(
    *,
    llm: Any,
    dataset_name: str,
    output_dir: Path,
    max_new_tokens: int,
    parallel: int | None,
    generation_only: bool = False,
    limit: int | None = None,
) -> None:
    """Batch-generate with vLLM, resume JSONL samples, and score with EvalPlus."""
    if dataset_name not in DATASET_STOPS:
        raise ValueError(f"Unsupported EvalPlus dataset: {dataset_name}")
    if limit is not None:
        if limit < 1:
            raise ValueError("EvalPlus limit must be positive")
        if not generation_only:
            raise ValueError("EvalPlus subsets can only be used with generation_only=True")

    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.sanitize import sanitize

    problems = get_human_eval_plus() if dataset_name == "humaneval" else get_mbpp_plus()
    selected = list(problems.items())[:limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"{dataset_name}.jsonl"
    completed: set[str] = set()
    if samples_path.is_file():
        with samples_path.open("r", encoding="utf-8") as handle:
            completed = {str(json.loads(line)["task_id"]) for line in handle if line.strip()}

    pending = [(task_id, problem) for task_id, problem in selected if task_id not in completed]
    if pending:
        prompts = [prepare_evalplus_prompt(problem["prompt"], dataset_name) for _, problem in pending]
        stops = COMMON_STOPS + DATASET_STOPS[dataset_name]
        completions = _generate_vllm(llm, prompts, max_new_tokens, stops)
        with samples_path.open("a", encoding="utf-8") as handle:
            for position, ((task_id, problem), prompt, completion) in enumerate(
                zip(pending, prompts, completions, strict=True), start=1,
            ):
                solution = sanitize(prompt + completion.replace("\t", "    "),
                                    entrypoint=problem["entry_point"])
                handle.write(json.dumps({"task_id": task_id, "solution": solution}) + "\n")
                handle.flush()
                print(f"[{dataset_name} {position}/{len(pending)} pending] {task_id}")

    print(f"Generated {samples_path}")
    if not generation_only:
        from evalplus.evaluate import evaluate
        evaluate(dataset=dataset_name, samples=str(samples_path), parallel=parallel)
