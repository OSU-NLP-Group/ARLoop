#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os

from loopedlm.evaluation.runtime import PROJECT_ROOT, configure_evaluation_runtime
from recpre.path_utils import resolve_project_path


DEFAULT_TASKS = "arloop_controlled"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a native AttnRes checkpoint with lm-evaluation-harness 0.4.12."
    )
    parser.add_argument("--checkpoint", default=None, help="Full Fabric checkpoint or raw model state dict.")
    parser.add_argument(
        "--config",
        default="configs/fineweb_edu/140m_arloop_full_r2.yaml",
        help="Training YAML used to construct the model.",
    )
    parser.add_argument("--tokenizer", default=None, help="Override tokenizer_path from the training YAML.")
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="Comma-separated lm-eval task/group names.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for equal-length loglikelihood requests.")
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-gen-toks", type=int, default=512)
    parser.add_argument(
        "--prepend-bos",
        action=argparse.BooleanOptionalAction, default=True,
        help="Prepend BOS to every prompt (required by the paper evaluation protocol).",
    )
    parser.add_argument("--num-fewshot", type=int, default=0)
    parser.add_argument("--limit", type=float, default=None, help="Smoke-test limit; omit for final metrics.")
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--compile", action="store_true", help="Compile the inference model with torch.compile.")
    parser.add_argument("--allow-code-execution", action="store_true", help="Required for MBPP/HumanEval execution.")
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=True, help="Allow dataset downloads (default); --no-online requires cached assets.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Download/cache selected lm-eval datasets, then exit without loading a checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="results/arloop_control.json",
        help="Result JSON path inside the repository.",
    )
    parser.add_argument("--log-samples", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = configure_evaluation_runtime(online=args.online)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unsafe_tasks = {"mbpp", "humaneval", "mbpp_plus", "humaneval_plus"}
    if unsafe_tasks.intersection(tasks) and not args.allow_code_execution:
        raise SystemExit(
            "MBPP/HumanEval execute generated Python. Re-run in an isolated, network-disabled "
            "environment with --allow-code-execution."
        )
    if args.allow_code_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Imports are deliberately delayed until cache/offline environment variables are set.
    from lm_eval import simple_evaluate, utils
    from lm_eval.tasks import TaskManager

    from loopedlm.evaluation.lm_eval_adapter import AttnResLMEval

    task_manager = TaskManager(include_path=PROJECT_ROOT / "lm_eval_tasks")
    if args.prepare_only:
        if not args.online:
            raise SystemExit("--prepare-only requires --online on the networked staging host")
        task_manager.load(tasks)
        print(f"Evaluation assets cached under {runtime}")
        return
    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --prepare-only is used")

    adapter = AttnResLMEval(
        config=args.config,
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_gen_toks=args.max_gen_toks,
        prepend_bos=args.prepend_bos,
        compile_model=args.compile,
    )
    result = simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        bootstrap_iters=args.bootstrap_iters,
        log_samples=args.log_samples,
        confirm_run_unsafe_code=args.allow_code_execution,
        task_manager=task_manager,
    )
    if result is None:
        raise RuntimeError("lm-eval returned no rank-zero result")

    output = resolve_project_path(args.output, label="evaluation output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, default=utils.handle_non_serializable)
        handle.write("\n")
    print(utils.make_table(result))
    if "groups" in result:
        print(utils.make_table(result, "groups"))
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
