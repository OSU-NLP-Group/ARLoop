"""AdamW and the parameter grouping used by the release recipes."""
from functools import partial
import torch


def get_param_groups(
    named_parameters,
    no_weight_decay_for_bias_and_norm_params=True,
    weight_lr_scale=1.0,
    no_wd_on_embedding=False,  # for tied models this needs to be false to have wd on the logit layer
    verbose=False,
):
    param_groups = []

    weights_group = []  # default group
    weights_no_wd_group = []  # weights with _no_weight_decay attribute
    embedding_group = []
    embedding_no_wd_group = []  # embeddings with _no_weight_decay attribute
    scale_and_norm_group = []

    for name, param in named_parameters:
        # Check if this parameter has _no_weight_decay attribute
        has_no_wd = getattr(param, "_no_weight_decay", False)

        # Strip FSDP/compile wrapper prefixes for matching
        clean_name = name
        for prefix in ["_orig_mod.", "_forward_module.", "_fsdp_wrapped_module."]:
            clean_name = clean_name.replace(prefix, "")
        lname = clean_name.lower()
        if "wte" in lname or "embedding" in lname or "abacus" in lname or "lm_head" in lname:
            if has_no_wd:
                embedding_no_wd_group.append(param)
                if verbose:
                    print(f"Matched {name} to embedding group (no weight decay via attribute)")
            else:
                embedding_group.append(param)
                if verbose:
                    print(f"Matched {name} to embedding group")
        elif "ln_f" in lname or "norm" in lname or "bias" in lname:
            scale_and_norm_group.append(param)
            if verbose:
                print(f"Matched {name} to scale and norm_group")
        elif (
            "attnres_queries" in lname
            or "attnres_query" in lname
            or "proj" in lname
            or "qkv" in lname
            or "fc" in lname
            or "mlp" in lname
            or param.ndim == 2
        ):
            if has_no_wd:
                weights_no_wd_group.append(param)
                if verbose:
                    print(f"Matched {name} to main weight group (no weight decay via attribute)")
            else:
                weights_group.append(param)
                if verbose:
                    print(f"Matched {name} to main weight group")
        elif has_no_wd:
            # Parameters with _no_weight_decay that don't match other patterns
            weights_no_wd_group.append(param)
            if verbose:
                print(f"Matched {name} to no-weight-decay group (via attribute)")
        else:
            raise ValueError(f"param {name} could not be matched to an optim group in recpre/optim.py")

    param_groups.append({"params": weights_group, "base_lr": weight_lr_scale})
    if weights_no_wd_group:
        param_groups.append({"params": weights_no_wd_group, "base_lr": weight_lr_scale, "weight_decay": 0.0})
    param_groups.append({"params": embedding_group, "base_lr": 1.0})
    if no_wd_on_embedding:
        param_groups[-1]["weight_decay"] = 0.0
    if embedding_no_wd_group:
        param_groups.append({"params": embedding_no_wd_group, "base_lr": 1.0, "weight_decay": 0.0})
    param_groups.append({"params": scale_and_norm_group, "base_lr": 1.0})
    if no_weight_decay_for_bias_and_norm_params:
        param_groups[-1]["weight_decay"] = 0.0

    return param_groups



def get_optimizer(optimizer_name: str, allow_fusion: bool = True):
    if optimizer_name != "AdamW":
        raise ValueError(f"Only AdamW is supported, got {optimizer_name!r}")
    return partial(torch.optim.AdamW, fused=True) if allow_fusion else torch.optim.AdamW
