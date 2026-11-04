from loopedlm.constants import DEFAULT_HF_ORG


configs = [
    dict(
        name="arloop-small-140m",
        hf_config=dict(org=DEFAULT_HF_ORG, name="arloop-small-140m"),
        # Context
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=768,
        n_layer=6,
        loop_count=4,
        attnres_block_size=2,
        use_value_embeddings=True,
        num_attention_heads=6,
        num_key_value_heads=6,
        intermediate_size=3072,
        bias=False,
        tie_embeddings=True,
        # Model class
        architecture_class_name="ARLoop",
        block_class_name="ARLoopBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=True,
        logit_softcap=None,
        # Initialization
        init_strategy="scaled-zero",
        init_orthogonal=True,
    ),
]
