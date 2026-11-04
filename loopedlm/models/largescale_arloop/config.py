from dataclasses import dataclass

from loopedlm.models.arloop.config import ARLoopConfig


@dataclass
class LargeScaleARLoopConfig(ARLoopConfig):
    """LargeScale SharedVE configuration using optimized FLA AttnRes."""

    attnres_sliding_window: bool = True
    sw_blocks: int = 3
    use_value_embeddings: bool = True
    cast_value_embeddings_to_residual_dtype: bool = True
    shared_value_embeddings: bool = True
    mlp_class_name: str = "GatedMLP"
    nonlin_name: str = "SiLU"
    model_class_name: str = "ARLoop"

    def __post_init__(self) -> None:
        if not self.use_value_embeddings:
            raise ValueError(
                "LargeScaleARLoop requires use_value_embeddings=True"
            )
        super().__post_init__()
