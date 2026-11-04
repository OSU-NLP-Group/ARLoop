#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os

from loopedlm.evaluation.runtime import PROJECT_ROOT, configure_evaluation_runtime
from recpre.path_utils import resolve_project_path
from loopedlm.vllm.source import add_source_arguments, model_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the ARLoop large-scale benchmark suite with a single-GPU vLLM engine."
    )
    add_source_arguments(parser)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-gen-toks", type=int, default=1024)
    parser.add_argument("--max-batch-size", type=int, default=64, help="Maximum concurrent sequences per GPU.")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--prepend-bos", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--evalplus-max-new-tokens", type=int, default=1024)
    parser.add_argument("--evalplus-parallel", type=int, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=100000)
    parser.add_argument("--limit", type=float, default=None, help="lm-eval smoke-test limit only.")
    parser.add_argument("--evalplus-limit", type=int, default=None)
    parser.add_argument("--log-samples", action="store_true")
    parser.add_argument("--online", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--generation-only", action="store_true", help="Do not execute EvalPlus samples.")
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default="results/arloop_all")
    return parser.parse_args()


def _print_plan() -> None:
    print("1. arloop_largescale: ARC-C 25-shot; HellaSwag 10-shot; WinoGrande 5-shot;")
    print("   OpenBookQA 0-shot; GSM8K CoT 5-shot; Minerva MATH 4-shot")
    print("2. EvalPlus: MBPP/MBPP+ and HumanEval/HumanEval+, 0-shot")


def main() -> None:
    args = parse_args()
    _print_plan()
    print(f"vLLM: single GPU, max concurrent sequences={args.max_batch_size}")
    if args.dry_run:
        return
    if not args.generation_only and not args.allow_code_execution:
        raise SystemExit(
            "EvalPlus executes generated Python. Use the isolated, network-disabled container and "
            "pass --allow-code-execution, or pass --generation-only."
        )

    configure_evaluation_runtime(online=args.online)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if args.allow_code_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    from lm_eval import simple_evaluate, utils
    from lm_eval.models.vllm_causallms import VLLM
    from lm_eval.tasks import TaskManager

    from loopedlm.evaluation.evalplus_adapter import run_evalplus_dataset
    from loopedlm.vllm import register

    register()
    output_dir = resolve_project_path(args.output_dir, label="vLLM evaluation output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    task_manager = TaskManager(include_path=PROJECT_ROOT / "lm_eval_tasks")
    with model_source(args) as source:
        adapter = VLLM(
            data_parallel_size=1,
            **source,
            dtype=args.dtype,
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
            tasks=["arloop_largescale"],
            num_fewshot=None,  # Preserve each task's shots from the group definition.
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            log_samples=args.log_samples,
            task_manager=task_manager,
        )
        if result is None:
            raise RuntimeError("lm-eval returned no rank-zero result")
        result_path = output_dir / "arloop_largescale.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False, default=utils.handle_non_serializable)
            handle.write("\n")
        print(utils.make_table(result))
        if "groups" in result:
            print(utils.make_table(result, "groups"))

        evalplus_dir = output_dir / "evalplus"
        evalplus_dir.mkdir(parents=True, exist_ok=True)
        for dataset_name in ("mbpp", "humaneval"):
            print(f"\n===== EvalPlus: {dataset_name} and {dataset_name}+ =====", flush=True)
            run_evalplus_dataset(
                llm=adapter.model,
                dataset_name=dataset_name,
                output_dir=evalplus_dir,
                max_new_tokens=args.evalplus_max_new_tokens,
                parallel=args.evalplus_parallel,
                generation_only=args.generation_only,
                limit=args.evalplus_limit,
            )

        print(f"\nAll vLLM benchmark stages completed under {output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
