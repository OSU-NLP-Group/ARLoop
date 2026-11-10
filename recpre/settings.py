import os
import json
import torch

from dataclasses import dataclass, field
from contextlib import nullcontext
from typing import Union, Optional, Any, Literal

from loopedlm.models.config import AnyConfig, Config as DynamicConfig
from recpre.path_utils import resolve_project_path

from transformers import AutoModelForCausalLM, AutoConfig


@dataclass
class HuggingfaceConfig:
    """need to properly merge HF one day"""

    name: str
    checkpoint: Optional[str]
    block_size: Optional[int] = None
    strategy: Optional[str] = None
    abacus_ids: list[int] = field(default_factory=lambda: list(range(10)))  # Will be initialized correctly later

    @property
    def Block(self):
        if "llama" in self.name.lower():
            from transformers.models.llama.modeling_llama import LlamaDecoderLayer

            return LlamaDecoderLayer
        else:
            raise ValueError("Provide the block name for this architecture.")

    def construct_model(self, objective, gradient_checkpointing: bool, **kwargs) -> torch.nn.Module:
        from axonn.models.transformers import parallelize

        source = self.checkpoint or self.name
        with parallelize(source) if self.strategy == "axonn_tp" else nullcontext():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(source))

        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        return model


@dataclass
class DataEntry:
    type: Literal["packed-bin", "stream"]
    prefix: str
    weight: int = 1
    data_dir: Optional[str] = None
    scheduler_ratios: Optional[list[Any] | tuple[Any, ...]] = None
    dataset: Optional[str] = None
    dataset_config: Optional[str] = None
    dataset_split: str = "train"
    text_column: str = "text"
    data_files: Optional[list[str]] = None
    packing_buffer_size: int = 1000
    shuffle_buffer_size: int = 0


@dataclass
class FabricConfig:
    optimize_communication: bool = False
    all_reduce_dtype: Optional[str] = None
    ddp_find_unused_parameters: bool = True
    ddp_static_graph: bool = False
    ddp_gradient_as_bucket_view: bool = False
    ddp_bucket_cap_mb: Optional[int] = None
    row_tensor_parallel_size: int = 1
    col_tensor_parallel_size: int = 1
    depth_tensor_parallel_size: int = 1


@dataclass
class CLISettings:
    # Main settings
    run_name: str = "default-run"  # The name for logging.
    out_dir: str = None  # type: ignore # The directory to save checkpoints. Required to be given or set as OUT_DIR
    resume: bool = True  # Whether to resume from a checkpoint in the out_dir.
    resume_checkpoint_path: Optional[str] = None  # Optional explicit path to checkpoint to resume from. If None, auto-searches.
    resume_model_only: bool = False  # Load only model weights from resume_checkpoint_path; keep new optimizer/data state.
    resume_data_state: bool = True  # Restore dataloader state when resuming.
    resume_wandb_run: bool = True  # Reuse the checkpoint's WandB run ID when resuming.
    resume_use_checkpoint_out_dir: bool = True  # Derive out_dir from resume_checkpoint_path for in-place resumes.
    max_tokens: int = 0  # The maximum number of tokens to train on (determines max_iters).
    max_steps: Optional[int] = None  # Set max_tokens=0 if setting max_steps
    max_flops: Optional[float] = None  # Alternative: specify training budget in FLOPs (derives max_tokens from model).
    token_to_param_ratio: Optional[float] = None  # Alternative: specify tokens as ratio of params (e.g., 7.0 for 7:1).
    exit_at_step: Optional[int] = None  # Stop training at this step (for testing). LR schedule still uses max_steps.
    seed: int = 1337  # The random seed to use for reproducibility.

    # Model configuration
    model_name: str = "tiny-llama-1.1b"  # The model name to use when creating the model from config.py / config_dynamic
    model_impl: str = "recpre"  # The model name to use when creating the model from config.py
    block_size: int = 2048  # The block size to use (lit-gpt-ese for sequence length).
    ignore_block_size_mismatch: bool = False  # Whether to ignore block size mismatch.
    model_checkpoint: Optional[str] = None  # The model checkpoint to load. Else, from config.

    # Training hyperparameters
    world_batch_size: int = 2048  # The total batch size across all devices and nodes.
    batch_size_ramp: int = 0  # Over how many mbs steps to linearly increase the batch size to world_batch_size
    optimizer: Literal["AdamW"] = "AdamW"
    optim_config: dict[str, Any] = field(
        default_factory=lambda: dict(
            lr=0.0004,  # The learning rate.
            weight_decay=0.1,  # The weight decay.
            betas=(0.9, 0.95),  # The beta parameters for the Adam optimizer.
            eps=1e-8,  # The eps parameter for the Adam optimizer
        )
    )
    grad_clip: float = 1.0  # The gradient clipping value.
    warmup_steps: float = 0  # Warmup steps: int for absolute count, float < 1 for percentage of max_steps.
    cooldown_steps: float = 0  # Cooldown steps: int for absolute count, float < 1 for percentage of max_steps.
    lr_schedule: str = "cosine"  # The learning rate schedule to use.
    min_lr: float = 0.00004  # The minimum learning rate to decay to.
    no_weight_decay_for_bias_and_norm_params: bool = False  # do not use weight decay for bias and norm params

    # Objective and Regularization
    # Ignore no vocabulary token by default, so tokenizer token 0 (SmolLM BOS/EOS)
    # remains supervised. Set this to a real pad id only when the data contains padding.
    loss_ignore_index: Optional[int] = -100
    label_smoothing: float = 0.0
    z_regularization: float = 0.0

    # Implementation and backend
    fabric_strategy: str = "ddp"  # The fabric strategy to use: ddp, fsdp, axonn_tp.
    fabric_precision: Literal["bf16-true", "bf16-mixed", "16-mixed", "16", "32"] = "bf16-mixed"
    fabric_use_lightning_environment: bool = False  # If False, use the auto setting, True, use LightningEnvironment.
    fabric: FabricConfig = field(
        default_factory=lambda: FabricConfig(
            optimize_communication=False,
            all_reduce_dtype=None,
            ddp_find_unused_parameters=True,
            ddp_static_graph=False,
            ddp_gradient_as_bucket_view=False,
            ddp_bucket_cap_mb=None,
            row_tensor_parallel_size=1,  # The size of the row tensor parallel dimension
            col_tensor_parallel_size=1,  # The size of the col tensor parallel dimension
            depth_tensor_parallel_size=1,  # The size of the depth tensor parallel dimension
        )
    )
    micro_batch_size: int = 4  # The micro batch size to use.
    compile_model: bool = False  # Whether to compile the model.
    compile_backend: str = "inductor"  # Also supports selective_pre_ddp and regional_attnres_pre_ddp.
    compile_mode: str = "default"  # Mode: "default", "reduce-overhead", "max-autotune"
    compile_fullgraph: bool = False  # Whether to require full graph compilation
    compile_dynamic: bool = False  # Whether to enable dynamic shapes
    matmul_precision: str = "high"  # enable tf32 acc on cuda with this
    dataloader_num_workers: int = 0  # The number of workers to use for the dataloaders.
    gradient_checkpointing: bool = False  # Whether to use activation checkpointing
    allow_nonfinite_loss: bool = False  # whether to end training immediately if non-finite loss is encountered
    compiled_autograd: bool = False
    compile_optimizer: bool = False
    drop_unused_tied_lm_head: bool = False
    dynamo_ddp_config: Literal["ddp_optimizer", "python_reducer", "no_optimization"] = "ddp_optimizer"
    skip_nonfinite_grads: bool = False
    fail_instead_of_recompile: bool = False  # code fails instead of recompiling
    # us this option to prevent dist jobs wasting time with cache failures

    # Logging
    logger_name: str = "wandb"
    logger_entity: str = ""
    logger_project: str = "arloop"
    logger_group: Optional[str] = None
    model_telemetry: bool = True  # Whether to monitor important model values to look for spikes. May increase overhead
    extreme_telemetry: bool = False  # Enable expensive metrics like adapter spectral norms. Only computed when model_telemetry is also True.
    shape_watching_steps: int = 3  # Number of iterations to watch shapes for. Set to 0 to disable.
    log_step_interval: int = 1  # The base interval for logging (scales with gradient_accumulation_steps).
    eval_iters: int = 100  # The number of iterations to process during a validation loop.
    eval_num_sequences: Optional[int] = None  # Fixed global validation rows; overrides eval_iters when set.
    eval_precision: Literal["same", "fp32", "bf16", "fp16", "bf16-mixed", "fp16-mixed"] = "same"  # Precision for evaluation. "same" uses training precision.
    cache_validation_batches: bool = True  # Persist the fixed validation prefix so later runs can reuse it.
    validation_batch_cache_dir: Optional[str] = None  # Defaults to the project data/validation_batch_cache directory.
    save_step_interval: int = 2000  # The number of microbatch steps between saving.
    save_wallclock_interval_seconds: Optional[float] = None  # Overrides step intervals; synchronized at optimizer boundaries.
    save_optimizer_step_interval: Optional[int] = None  # If set, save on this optimizer-step interval instead.
    eval_step_interval: int = 2000  # Optimizer steps between validation runs.
    save_first_step: bool = False  # Whether to save the checkpoint at the first step
    save_last_step: bool = False  # Whether to save the checkpoint at the last step
    save_n_min_before_job_done: Optional[int] = None  # Save the checkpoint n minutes before current job done
    max_checkpoints_to_keep: Optional[int] = None  # Keep only the newest N checkpoint steps for this run.
    measure_utilization: bool = True  # Print FLOPs and MFU

    # Data Handling
    # PKDS arguments:
    shuffle_filenames: bool = True  # Shuffle packed source filenames.
    shuffle_blocks: bool = True  # Shuffle rows in the global blended reader.
    packed_sequence_shuffle_chunk_size: int = 1  # (Blended packed-bin only.) Shuffle chunks of this many rows; 1 is a full row permutation.
    data_config: dict[str, list[DataEntry]] = field(
        default_factory=lambda: {
            "train_data": [DataEntry("packed-bin", "", 1)],
            "val_data": [DataEntry("packed-bin", "", 1)],
        }
    )
    # The path to the tokenizer to use [required to identify pad_token_id even for pkds]
    tokenizer_path: str = "tokenizer/parcae_tokenizer"
    model_config: Union[AnyConfig, HuggingfaceConfig] = field(init=False)
    model_overwrite: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Validate arguments
        if self.out_dir is None:
            self.out_dir = os.getenv("OUTPUT_DIR", "NOT_FOUND")
        assert self.out_dir != "NOT_FOUND"
        self.out_dir = str(resolve_project_path(self.out_dir, label="out_dir"))
        assert self.tokenizer_path, "Tokenizer has to be specified."

        # If data_config is a string, load it from a file.
        self.data_config = self._validate_data_config()
        self._expand_paths()

        # Tensor parallelism is implemented by the AxoNN fabric only.
        if (
            self.fabric.depth_tensor_parallel_size > 1
            or self.fabric.row_tensor_parallel_size > 1
            or self.fabric.col_tensor_parallel_size > 1
        ):
            assert self.fabric_strategy == "axonn_tp", "x_tensor_parallel_size > 1 implies use of axonn_tp."

        self._parse_environment_variables()

        # Add any derived cfg here
        self.node_batch_size = self.world_batch_size // self.num_nodes
        self.loader_block_size = self.block_size + 1
        self.global_total_time = 0
        self.max_tokens_per_device = 0
        self.tokens_per_step = 0

        self.batch_size = self.node_batch_size // self.devices
        if self.batch_size_ramp == 0:
            self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size
        else:
            self.gradient_accumulation_steps = 1
        self.replicas = self.devices * self.num_nodes

        self.dataset_names = [i.prefix for i in self.data_config["train_data"]]
        self.train_dataset_prefixes = [ds.prefix for ds in self.data_config["train_data"]]
        self.val_dataset_prefixes = (
            [ds.prefix for ds in self.data_config["val_data"]] if "val_data" in self.data_config else []
        )

        self._validate_args()

        # Finally, store model config object itself
        self.model_config = DynamicConfig.from_name(self.model_name, **self.model_overwrite)

        # Set strategy
        self.model_config.strategy = self.fabric_strategy

    def _validate_args(self):
        assert self.eval_step_interval > 0
        # Allow one of: max_steps, max_tokens, max_flops, or token_to_param_ratio
        specified = sum([
            self.max_steps is not None,
            self.max_tokens > 0,
            self.max_flops is not None,
            self.token_to_param_ratio is not None,
        ])
        assert specified == 1, (
            f"Exactly one of max_steps, max_tokens, max_flops, or token_to_param_ratio must be specified"
        )
        assert len(set(self.dataset_names)) == len(self.data_config["train_data"]), (
            "please provide different names for each subset"
        )

        # Any additional sanity checks here.
        assert self.gradient_accumulation_steps > 0, "derived gradient_accumulation_steps must be > 0"
        assert self.packed_sequence_shuffle_chunk_size > 0, (
            "packed_sequence_shuffle_chunk_size must be positive"
        )
        if self.max_checkpoints_to_keep is not None:
            assert (
                isinstance(self.max_checkpoints_to_keep, int)
                and not isinstance(self.max_checkpoints_to_keep, bool)
                and self.max_checkpoints_to_keep > 0
            ), "max_checkpoints_to_keep must be a positive integer or null"
        if self.save_wallclock_interval_seconds is not None:
            assert self.save_wallclock_interval_seconds > 0, "save_wallclock_interval_seconds must be positive"
        if self.save_optimizer_step_interval is not None:
            assert (
                isinstance(self.save_optimizer_step_interval, int)
                and not isinstance(self.save_optimizer_step_interval, bool)
                and self.save_optimizer_step_interval > 0
            ), "save_optimizer_step_interval must be a positive integer or null"
        uses_stateful_rows = any(
            entry.type in {"packed-bin", "stream"}
            for entries in self.data_config.values()
            for entry in entries
        )
        if uses_stateful_rows:
            assert self.dataloader_num_workers == 0, (
                "Training data currently requires dataloader_num_workers=0 so each "
                "DDP rank has exactly one stateful iterator"
            )
        if self.eval_num_sequences is not None:
            assert self.eval_num_sequences > 0, "eval_num_sequences must be positive"
        if self.batch_size_ramp == 0:
            assert (
                self.world_batch_size
                == self.micro_batch_size * self.gradient_accumulation_steps * self.devices * self.num_nodes
            ), "world batch size should be: micro_batch_size * gradient_accumulation_steps * devices * num_nodes"
        else:
            assert self.world_batch_size % (self.micro_batch_size * self.devices * self.num_nodes) == 0

    def _expand_paths(self):
        for entries in self.data_config.values():
            for entry in entries:
                if entry.type == "stream":
                    if not entry.dataset:
                        raise ValueError("stream requires a Hugging Face dataset name")
                    if entry.data_dir or entry.scheduler_ratios is not None:
                        raise ValueError("stream does not use data_dir or a mixture scheduler")
                    if entry.packing_buffer_size < 1 or entry.shuffle_buffer_size < 0:
                        raise ValueError("Invalid streaming buffer size")
                    continue
                if not entry.data_dir:
                    raise ValueError(f"data_dir is required for data source {entry.prefix!r}")
                if entry.data_dir is not None:
                    entry.data_dir = os.path.expandvars(entry.data_dir)

    def _parse_environment_variables(self):
        """Parse env variables and directly store as non-field attributes"""
        self.SLURM_JOB_ID = int(os.getenv("SLURM_JOB_ID", 0))
        self.SLURM_ARRAY_JOB_ID = int(os.getenv("SLURM_ARRAY_JOB_ID", 0))
        self.SLURM_ARRAY_TASK_ID = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
        self.SLURM_ARRAY_TASK_COUNT = int(os.getenv("SLURM_ARRAY_TASK_COUNT", 1))
        self.MASTER_ADDR = os.getenv("MASTER_ADDR", "0")
        self.MASTER_PORT = int(os.getenv("MASTER_PORT", 0))
        self.WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        self.RANK = int(os.getenv("SLURM_PROCID", "0"))
        self.devices = int(
            os.getenv(
                "ARLOOP_DEVICES",
                os.getenv("SLURM_NTASKS_PER_NODE", os.getenv("LOCAL_WORLD_SIZE", torch.cuda.device_count())),
            )
        )
        self.num_nodes = int(os.getenv("ARLOOP_NUM_NODES", os.getenv("SLURM_JOB_NUM_NODES", 1)))

    def _validate_data_config(self) -> dict[str, list[DataEntry]]:
        if isinstance(self.data_config, str):
            try:
                with open(self.data_config, mode="r") as json_file:
                    self.data_config = json.load(json_file)
            except Exception as e:
                raise ValueError(
                    f"data_config passed was a string, but failed to load as a json object from {self.data_config}: {e}"
                )
        return self.data_config
