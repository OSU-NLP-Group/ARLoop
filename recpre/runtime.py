"""Torch backend setup and distributed runtime helpers."""
import os
import re
import time
import torch
from loopedlm.attention_backends.flash_attention import should_enable_cudnn_sdpa

def set_torch_flags(cfg):
    torch.set_float32_matmul_precision(cfg.matmul_precision)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    try:
        # Blackwell falls back from the SM90-only FA3 extension to PyTorch
        # SDPA. Keep Flash SDPA enabled above and also let PyTorch consider its
        # Blackwell-optimized cuDNN SDPA backend.
        torch.backends.cuda.enable_cudnn_sdp(should_enable_cudnn_sdpa())
    except AttributeError:
        pass

    torch._dynamo.config.optimize_ddp = cfg.dynamo_ddp_config
    torch._dynamo.config.compiled_autograd = cfg.compiled_autograd

    if cfg.compile_model:
        import torch._inductor.config as inductor_config
        import torch._functorch.config as functorch_config
        inductor_config.emulate_precision_casts = True
        inductor_config.triton.cudagraphs = False
        inductor_config.coordinate_descent_tuning = False
        inductor_config.fx_graph_cache = False
        functorch_config.donated_buffer = False
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = False

    if cfg.fail_instead_of_recompile:
        torch._dynamo.config.error_on_recompile = True



def derive_precision(precision, strategy_details):
    import torch.distributed.fsdp

    param_dtype = torch.bfloat16 if "bf16" in precision else torch.float16 if "16" in precision else torch.float32
    reduce_dtype = torch.float32 if "mixed" in precision else param_dtype
    if (r := strategy_details.all_reduce_dtype) is not None:
        reduce_dtype = (
            torch.float16
            if r in ["16", "fp16", "fp16-mixed"]
            else torch.bfloat16
            if r in ["bf16", "bf16-mixed"]
            else torch.float32
        )
    return torch.distributed.fsdp.MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=torch.float32,
        keep_low_precision_grads=False,
        # cast_forward_inputs=False,
    )



def warmup_or_early_fail_allreduce(fabric):
    if torch.distributed.is_initialized():
        fabric.print("Staging allreduce warmup")
        device = fabric.device
        # Creating random data for warmup
        flat_params = torch.randn(128 * 1024 * 1024 // 4, device=device)
        num_stages = 8
        chunk_size = flat_params.numel() // num_stages

        for i in range(num_stages):
            end = min((i + 1) * chunk_size, flat_params.numel())
            chunk = flat_params[:end]
            torch.distributed.all_reduce(chunk)
            torch.cuda.current_stream().synchronize()  # Force completion
            fabric.print(f"Warmup stage {i} [{chunk.numel() // (1024 * 1024 // 4)} MB] really completed")

        torch.distributed.barrier()
        fabric.print(f"{time.ctime()[:-5]}: All warmup stages passed")



def _get_time_from_slurm() -> int:
    try:
        global_total_str_parse = os.popen("squeue -h -j $SLURM_JOBID -o %L").read()  # this is slow
        global_total_str_parse = global_total_str_parse.strip("\n")
        global_total_str_parse = [int(i) for i in re.split(":|-", global_total_str_parse)]
        if len(global_total_str_parse) == 4:
            global_total_time = (
                24 * 3600 * global_total_str_parse[0]
                + 3600 * global_total_str_parse[1]
                + 60 * global_total_str_parse[2]
                + global_total_str_parse[3]
            )
        elif len(global_total_str_parse) == 3:
            global_total_time = (
                3600 * global_total_str_parse[0] + 60 * global_total_str_parse[1] + global_total_str_parse[2]
            )
        elif len(global_total_str_parse) == 2:
            global_total_time = 60 * global_total_str_parse[0] + global_total_str_parse[1]
    except Exception as e:
        print(e)
        global_total_time = 9999999999999999
    return global_total_time
