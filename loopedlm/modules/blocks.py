from typing import Optional

import torch
from torch import Tensor

from loopedlm.models.config import AnyConfig
from loopedlm.modules.mixer import CausalSelfAttention


class TransformerPreNormBlock(torch.nn.Module):
    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        x = self.attn(self.norm_1(x), freqs_cis, mask, **kwargs) + x
        x = self.mlp(self.norm_2(x)) + x
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_2, "normalization")


class TransformerPostResidualNormBlock(torch.nn.Module):
    """Transformer block with norm after each residual addition.

    The update order is:
        x = norm(x + attn(norm(x)))
        x = norm(x + mlp(norm(x)))
    """

    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.norm_1 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.norm_1_post = config.Norm(config.n_embd, eps=config.norm_eps)
        self.norm_2 = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.norm_2_post = config.Norm(config.n_embd, eps=config.norm_eps)
        self.layer_id = layer_id

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        x = self.norm_1_post(x + self.attn(self.norm_1(x), freqs_cis, mask, **kwargs))
        x = self.norm_2_post(x + self.mlp(self.norm_2(x)))
        return x

    def reset_parameters(self) -> None:
        self.config.init.apply(self.norm_1, "normalization")
        self.config.init.apply(self.norm_1_post, "normalization")
        self.config.init.apply(self.norm_2, "normalization")
        self.config.init.apply(self.norm_2_post, "normalization")


class ARLoopBlock(torch.nn.Module):
    """Attention Residual block without built-in residual additions.

    The owning model computes Block AttnRes inputs and partial-block residual
    accumulation. This block only owns the attention/MLP sublayers and their
    pre-sublayer norms.
    """

    expanded = False

    def __init__(self, config: AnyConfig, layer_id: int) -> None:
        super().__init__()
        self.config = config
        self.attn_norm = config.Norm(config.n_embd, eps=config.norm_eps)
        self.attn = CausalSelfAttention(config, layer_id=layer_id)
        self.mlp_norm = config.Norm(config.n_embd, eps=config.norm_eps)
        self.mlp = config.MLP(config, layer_id=layer_id)
        self.layer_id = layer_id

    def forward_attention(self, h: Tensor, freqs_cis: Tensor, mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.attn(self.attn_norm(h), freqs_cis, mask, **kwargs)

    def forward_mlp(self, h: Tensor) -> Tensor:
        return self.mlp(self.mlp_norm(h))

    def forward(
        self,
        h: Tensor,
        freqs_cis: Tensor,
        mask: Optional[Tensor] = None,
        attnres_mode: str = "attn",
        **kwargs,
    ) -> Tensor:
        if attnres_mode == "attn":
            return self.forward_attention(h, freqs_cis, mask, **kwargs)
        if attnres_mode == "mlp":
            return self.forward_mlp(h)
        raise ValueError(f"Unsupported attnres_mode: {attnres_mode}")

    def reset_parameters(self) -> None:
        self.config.init.apply(self.attn_norm, "normalization")
        self.config.init.apply(self.mlp_norm, "normalization")
