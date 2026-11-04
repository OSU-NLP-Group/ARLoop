from loopedlm.models.gpt.init import GPTInit


class ARLoopInit(GPTInit):
    """ARLoop uses GPT per-layer initialization scaled to its loop depth."""
