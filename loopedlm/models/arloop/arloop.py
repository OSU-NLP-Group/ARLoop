from typing import Any, Optional

import torch

from loopedlm.models.arloop.fla_softmax import (
    AttnResFLAIndexedOwnerBackward,
)
from loopedlm.models.arloop.config import (
    ARLoopConfig,
)


class ARLoop(AttnResFLAIndexedOwnerBackward):
    """Unified looped transformer with the optimized FLA AttnRes path.

    Forward uses FLA's fused softmax routing and output RMSNorm. Backward
    defaults to indexed source-gradient accumulation with fused owner
    finalization. Standard FLA backward supports activation checkpointing.
    """

    def __init__(
        self,
        config: ARLoopConfig,
        objective: Optional[dict[str, Any]] = None,
        gradient_checkpointing: bool = False,
    ) -> None:
        if config.attnres_block_size < 2:
            raise ValueError("FLA ARLoop requires an even attnres_block_size >= 2")
        if not config.cast_value_embeddings_to_residual_dtype:
            raise ValueError("FLA ARLoop requires BF16 value-embedding casts")
        super().__init__(
            config,
            objective=objective,
            gradient_checkpointing=gradient_checkpointing,
        )

        if not config.shared_value_embeddings:
            return

        value_embedding_layers = tuple(self.value_embeds.keys())
        if not value_embedding_layers:
            raise ValueError(
                "ARLoop requires at least one VE layer"
            )

        shared_embedding = self.value_embeds[value_embedding_layers[0]]
        for layer_idx in value_embedding_layers[1:]:
            self.value_embeds[layer_idx] = shared_embedding
        self.shared_value_embedding_layers = value_embedding_layers

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Keep embedding parameters in FP32 and residual storage in BF16."""
        return super()._embed_tokens(input_ids).to(dtype=torch.bfloat16)
