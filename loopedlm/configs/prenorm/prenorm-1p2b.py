from loopedlm.constants import DEFAULT_HF_ORG


configs = [
    dict(
        name="prenorm-1p2b",
        hf_config=dict(org=DEFAULT_HF_ORG, name="prenorm-1p2b"),
        block_size=4096,
        vocab_size=49152,
        padding_multiple=64,
        n_embd=2048,
        n_layer=24,
        loop_count=4,
        use_value_embeddings=True,
        shared_value_embeddings=True,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=5632,
        bias=False,
        tie_embeddings=True,
        architecture_class_name="PreNorm",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="GatedMLP",
        nonlin_name="SiLU",
        qk_norm=True,
        logit_softcap=None,
        init_strategy="scaled",
        init_orthogonal=False,
    ),
]
