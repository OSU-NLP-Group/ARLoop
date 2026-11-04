from loopedlm.constants import DEFAULT_HF_ORG


def _config(
    scale: str,
    *,
    n_embd: int,
    n_layer: int,
    intermediate_size: int,
    attnres_block_size: int,
) -> dict:
    name = f"largescale-arloop-{scale}"
    return dict(
        name=name,
        hf_config=dict(org=DEFAULT_HF_ORG, name=name),
        block_size=4096,
        vocab_size=49152,
        padding_multiple=64,
        n_embd=n_embd,
        n_layer=n_layer,
        loop_count=1,
        attnres_block_size=attnres_block_size,
        attnres_sliding_window=True,
        sw_blocks=3,
        use_value_embeddings=True,
        cast_value_embeddings_to_residual_dtype=True,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=intermediate_size,
        bias=False,
        tie_embeddings=True,
        architecture_class_name="LargeScaleARLoop",
        block_class_name="ARLoopBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        qk_norm=True,
        logit_softcap=None,
        init_strategy="scaled",
        init_orthogonal=False,
    )


configs = [
    _config(
        "370m",
        n_embd=1536,
        n_layer=12,
        intermediate_size=4096,
        attnres_block_size=12,
    ),
    _config(
        "630m",
        n_embd=1536,
        n_layer=18,
        intermediate_size=4608,
        attnres_block_size=12,
    ),
    _config(
        "0p6b",
        n_embd=1536,
        n_layer=18,
        intermediate_size=4608,
        attnres_block_size=18,
    ),
    _config(
        "1p2b",
        n_embd=2048,
        n_layer=24,
        intermediate_size=5632,
        attnres_block_size=24,
    ),
]
