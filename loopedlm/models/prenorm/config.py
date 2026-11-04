from dataclasses import dataclass
import torch
from loopedlm.models.gpt.config import GPTConfig
from loopedlm.models.prenorm.init import PreNormInit


@dataclass
class PreNormConfig(GPTConfig):
    """Fixed-depth looped pre-norm transformer baseline."""

    loop_count: int = 1
    use_value_embeddings: bool = True
    shared_value_embeddings: bool = False
    model_class_name: str = "PreNorm"
    block_class_name: str = "TransformerPreNormBlock"

    def __post_init__(self):
        super().__post_init__()
        if self.shared_value_embeddings and not self.use_value_embeddings:
            raise ValueError("shared_value_embeddings requires use_value_embeddings=True")
        if self.loop_count < 1:
            raise ValueError(f"loop_count must be >= 1, got {self.loop_count}")
        self.init = PreNormInit(
            self.init_strategy,
            self.n_embd,
            self.intermediate_size,
            self.head_size,
            self.n_layer * self.loop_count,
            self.mup_model_scaling_factor,
            orthogonal=self.init_orthogonal,
            verbose=False,
            skip_reinitializing=self.skip_initialization,
        )

    def construct_model(self, **kwargs) -> torch.nn.Module:
        from loopedlm.models.prenorm import PreNorm

        return PreNorm(self, **kwargs)
