from loopedlm.models.gpt.init import GPTInit


class PreNormInit(GPTInit):
    """PreNorm uses GPT per-block initialization scaled to its loop depth."""
