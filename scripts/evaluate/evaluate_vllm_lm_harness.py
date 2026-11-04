#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os

from loopedlm.evaluation.runtime import PROJECT_ROOT, configure_evaluation_runtime
from recpre.path_utils import resolve_project_path
from loopedlm.vllm.source import add_source_arguments, model_source


DEFAULT_TASKS = "arloop_largescale"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an ARLoop HF model or native checkpoint with lm-eval and vLLM continuous batching."
    )
    add_source_arguments(parser)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-toks", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum concurrent sequences per GPU.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--prepend-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=float, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output", default="results/arloop_largescale.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_evaluation_runtime(online=args.online)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    unsafe_tasks = {"mbpp", "humaneval", "mbpp_plus", "humaneval_plus"}
    if unsafe_tasks.intersection(tasks) and not args.allow_code_execution:
        raise SystemExit("MBPP/HumanEval execute generated Python; pass --allow-code-execution in an isolated, network-disabled environment.")
    if args.allow_code_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    from lm_eval import simple_evaluate, utils
    from lm_eval.tasks import TaskManager

    from lm_eval.models.vllm_causallms import VLLM

    from loopedlm.vllm import register

    register()
    with model_source(args) as source:
        adapter = VLLM(
            **source,
            dtype=args.dtype,
            data_parallel_size=1,
            tensor_parallel_size=1,
            max_model_len=args.max_model_len,
            max_gen_toks=args.max_gen_toks,
            batch_size="auto",
            max_batch_size=args.max_batch_size,
            add_bos_token=args.prepend_bos,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            seed=0,
        )
        result = simple_evaluate(
            model=adapter,
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size="auto",
            max_batch_size=args.max_batch_size,
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            log_samples=args.log_samples,
            confirm_run_unsafe_code=args.allow_code_execution,
            task_manager=TaskManager(include_path=PROJECT_ROOT / "lm_eval_tasks"),
        )
        if result is None:
            raise RuntimeError("lm-eval returned no rank-zero result")
        output = resolve_project_path(args.output, label="vLLM evaluation output")
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
