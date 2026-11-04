from dataclasses import dataclass
from typing import Literal, Optional

import torch

from loopedlm.models.gpt.config import GPTConfig
from loopedlm.models.arloop.init import ARLoopInit


@dataclass
class ARLoopConfig(GPTConfig):
    """Looped GPT-style transformer with Block Attention Residuals."""

    attnres_backend: Literal["indexed", "fla"] = "indexed"
    loop_count: int = 1
    attnres_block_size: int = 2
    attnres_sliding_window: bool = False
    sw_blocks: Optional[int] = None
    use_value_embeddings: bool = True
    shared_value_embeddings: bool = False
    cast_value_embeddings_to_residual_dtype: bool = True
    model_class_name: str = "ARLoop"
    block_class_name: str = "ARLoopBlock"

    def __post_init__(self):
        super().__post_init__()
        if self.attnres_backend not in {"indexed", "fla"}:
            raise ValueError("attnres_backend must be indexed or fla")
        if self.loop_count < 1:
            raise ValueError(f"loop_count must be >= 1, got {self.loop_count}")
        if self.attnres_block_size < 1:
            raise ValueError(f"attnres_block_size must be >= 1, got {self.attnres_block_size}")
        if self.attnres_block_size != 1 and self.attnres_block_size % 2 != 0:
            raise ValueError("attnres_block_size must be 1 for full mode or even for block mode")
        if self.sw_blocks is not None:
            if isinstance(self.sw_blocks, bool) or not isinstance(self.sw_blocks, int):
                raise ValueError(f"sw_blocks must be an integer or None, got {self.sw_blocks!r}")
            if self.sw_blocks < 1:
                raise ValueError(f"sw_blocks must be >= 1, got {self.sw_blocks}")
            if not self.attnres_sliding_window:
                raise ValueError("sw_blocks requires attnres_sliding_window")
        if self.attnres_sliding_window:
            if self.model_class_name not in {
                "ARLoop",
                "LargeScaleARLoop",
            }:
                raise ValueError(
                    "attnres_sliding_window is currently supported only for ARLoop "
                    "and LargeScaleARLoop variants, "
                    f"got {self.model_class_name}"
                )
            if self.attnres_block_size == 1:
                raise ValueError("attnres_sliding_window does not support full AttnRes mode")
            layers_per_attnres_block = self.attnres_block_size // 2
            if self.n_layer % layers_per_attnres_block != 0:
                raise ValueError(
                    "attnres_sliding_window requires n_layer to be divisible by "
                    "attnres_block_size // 2, "
                    f"got n_layer={self.n_layer}, attnres_block_size={self.attnres_block_size}"
                )

        effective_depth = self.n_layer * self.loop_count
        self.init = ARLoopInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            effective_depth,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    @property
    def effective_sublayers(self) -> int:
        return self.loop_count * self.n_layer * 2

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from loopedlm.models.arloop import ARLoop

        return ARLoop(self, **kwargs)
