from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model

from loopedlm.evaluation.checkpoint import load_checkpoint


LOGGER = logging.getLogger(__name__)


@register_model("attnres_lt")
class AttnResLMEval(TemplateLM):
    """lm-evaluation-harness adapter for native AttnRes training checkpoints."""

    backend = "causal"

    def __init__(
        self,
        config: str,
        checkpoint: str,
        tokenizer: str | None = None,
        device: str = "cuda",
        dtype: str = "auto",
        batch_size: int | str = 1,
        max_length: int | None = None,
        max_gen_toks: int = 512,
        prepend_bos: bool = False,
        compile_model: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        if batch_size == "auto":
            raise ValueError("AttnResLMEval requires an explicit integer batch_size")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        loaded = load_checkpoint(
            config_path=config,
            checkpoint_path=checkpoint,
            tokenizer_path=tokenizer,
            device=device,
            dtype=dtype,
            compile_model=compile_model,
        )
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        self.model_config = loaded.config
        self.checkpoint_path = loaded.checkpoint_path
        self._device = torch.device(device)
        self.compute_dtype = loaded.compute_dtype
        self.prepend_bos = bool(prepend_bos)
        self.max_length = int(max_length or loaded.config.block_size)
        if self.max_length > loaded.config.block_size:
            raise ValueError(
                f"max_length={self.max_length} exceeds model block_size={loaded.config.block_size}"
            )
        self.max_gen_toks = int(max_gen_toks)
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            try:
                from loopedlm.attention_backends.flash_attention import should_enable_cudnn_sdpa

                torch.backends.cuda.enable_cudnn_sdp(should_enable_cudnn_sdpa())
            except AttributeError:
                pass

    @property
    def eot_token_id(self) -> int:
        if self.tokenizer.eos_id is not None:
            return int(self.tokenizer.eos_id)
        if self.tokenizer.bos_id is not None:
            return int(self.tokenizer.bos_id)
        raise ValueError("Tokenizer defines neither EOS nor BOS; an EOT/prefix token is required")

    @property
    def tokenizer_name(self) -> str:
        return str(self.tokenizer.checkpoint_dir or "attnres-lt-tokenizer")

    def tok_encode(
        self,
        string: str,
        add_special_tokens: bool | None = None,
        left_truncate_len: int | None = None,
        **_: Any,
    ) -> list[int]:
        # Training used add_bos=false/add_eos=false. Only TemplateLM's empty
        # context path explicitly inserts prefix_token_id.
        if add_special_tokens not in {None, False}:
            raise ValueError("This base-model adapter does not add tokenizer special tokens")
        result = self.tokenizer.encode(string, return_tensors=False)
        if self.prepend_bos:
            if self.tokenizer.bos_id is None:
                raise ValueError("--prepend-bos requires a tokenizer with a BOS token")
            bos_id = int(self.tokenizer.bos_id)
            if not result or result[0] != bos_id:
                result = [bos_id, *result]
        if left_truncate_len is not None:
            result = result[-left_truncate_len:]
        return result

    def tok_decode(self, tokens, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(list(tokens) if not isinstance(tokens, int) else [tokens], skip_special_tokens)

    @torch.inference_mode()
    def _model_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype != torch.float32,
        ):
            outputs = self.model(input_ids=input_ids, return_logits=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
        if logits is None:
            raise RuntimeError("Model returned no logits; evaluation requires the PyTorch LM head")
        return logits

    def _loglikelihood_tokens(
        self,
        requests: list[tuple[tuple[str, str] | None, list[int], list[int]]],
        disable_tqdm: bool = False,
        **_: Any,
    ) -> list[tuple[float, bool]]:
        prepared: list[tuple[int, tuple[str, str] | None, list[int], list[int], list[int]]] = []
        for index, (cache_key, context, continuation) in enumerate(requests):
            if not context or not continuation:
                raise ValueError("loglikelihood requires non-empty encoded context and continuation")
            if len(continuation) > self.max_length:
                raise ValueError(
                    f"Continuation has {len(continuation)} tokens, exceeding max_length={self.max_length}"
                )
            combined = (context + continuation)[-(self.max_length + 1) :]
            inputs = combined[:-1]
            scored = combined[-len(continuation) :]
            prepared.append((index, cache_key, inputs, scored, continuation))

        # Attention masks are intentionally not used by the native model. Batch
        # only equal-length sequences so padding cannot change logits.
        by_length: dict[int, list[tuple[int, tuple[str, str] | None, list[int], list[int], list[int]]]] = defaultdict(list)
        for item in prepared:
            by_length[len(item[2])].append(item)

        answers: list[tuple[float, bool] | None] = [None] * len(requests)
        total_batches = sum((len(group) + self.batch_size - 1) // self.batch_size for group in by_length.values())
        progress = tqdm(total=total_batches, disable=disable_tqdm, desc="AttnRes loglikelihood batches")
        for _, group in sorted(by_length.items(), reverse=True):
            for start in range(0, len(group), self.batch_size):
                chunk = group[start : start + self.batch_size]
                input_ids = torch.tensor([item[2] for item in chunk], dtype=torch.long, device=self.device)
                logits = self._model_logits(input_ids)
                for row, (index, cache_key, inputs, scored, original_continuation) in enumerate(chunk):
                    cont_len = len(scored)
                    selected = logits[row, len(inputs) - cont_len : len(inputs), :]
                    target = torch.tensor(scored, dtype=torch.long, device=self.device)
                    log_probs = F.log_softmax(selected.float(), dim=-1)
                    score = float(log_probs.gather(-1, target[:, None]).sum().item())
                    # A continuation that was left-truncated cannot be an exact
                    # greedy match to the original full continuation.
                    greedy = bool(torch.equal(selected.argmax(dim=-1), target)) and len(scored) == len(original_continuation)
                    answer = (score, greedy)
                    answers[index] = answer
                    if cache_key is not None:
                        self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                progress.update(1)
        progress.close()
        if any(answer is None for answer in answers):
            raise RuntimeError("Internal error: a loglikelihood request was not evaluated")
        return [answer for answer in answers if answer is not None]

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False) -> list[float]:
        results: list[float] = []
        for (text,) in tqdm(
            [request.args for request in requests],
            disable=disable_tqdm,
            desc="Tokenizing rolling loglikelihood",
        ):
            windows = [
                (None, context, continuation)
                for context, continuation in map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(text),
                        prefix_token=self.prefix_token_id,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            ]
            score = sum(value for value, _ in self._loglikelihood_tokens(windows, disable_tqdm=True))
            results.append(score)
            self.cache_hook.add_partial("loglikelihood_rolling", (text,), score)
        return results

    @torch.inference_mode()
    def generate_text(
        self,
        context: str,
        *,
        max_new_tokens: int,
        until: list[str] | None = None,
        temperature: float = 0.0,
        top_k: int | None = None,
        top_p: float | None = None,
        do_sample: bool = False,
    ) -> str:
        if max_new_tokens < 1:
            return ""
        max_context = self.max_length - max_new_tokens
        if max_context < 1:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} leaves no context within max_length={self.max_length}"
            )
        prompt = self.tok_encode(context)[-max_context:]
        if not prompt:
            prompt = [self.prefix_token_id]

        input_ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        cache = self.model.create_cache()
        generated: list[int] = []
        current = input_ids
        stop_strings = [item for item in (until or []) if item]

        for _ in range(max_new_tokens):
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.compute_dtype,
                enabled=self.compute_dtype != torch.float32,
            ):
                outputs = self.model.forward_for_generation(current, past_key_values=cache)
            logits = outputs["logits"][:, -1, :].float()
            if do_sample:
                if temperature <= 0:
                    raise ValueError("Sampling requires temperature > 0")
                logits = logits / temperature
                if top_k is not None and top_k > 0:
                    cutoff = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    mask = torch.zeros_like(remove).scatter(1, sorted_indices, remove)
                    logits = logits.masked_fill(mask, float("-inf"))
                next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
            else:
                next_token = int(logits.argmax(dim=-1).item())
            if next_token == self.eot_token_id:
                break
            generated.append(next_token)
            text = self.tok_decode(generated, skip_special_tokens=True)
            matches = [text.find(stop) for stop in stop_strings if stop in text]
            if matches:
                return text[: min(matches)]
            current = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
        return self.tok_decode(generated, skip_special_tokens=True)

    def generate_until(self, requests, disable_tqdm: bool = False) -> list[str]:
        results: list[str] = []
        for request in tqdm(requests, disable=disable_tqdm, desc="AttnRes generation"):
            context, raw_kwargs = request.args
            kwargs = dict(raw_kwargs)
            until = kwargs.pop("until", None)
            if until is None:
                stop = []
            elif isinstance(until, str):
                stop = [until]
            else:
                stop = list(until)
            max_new_tokens = int(kwargs.pop("max_gen_toks", kwargs.pop("max_new_tokens", self.max_gen_toks)))
            temperature = float(kwargs.pop("temperature", 0.0) or 0.0)
            do_sample = bool(kwargs.pop("do_sample", temperature > 0))
            top_k = kwargs.pop("top_k", None)
            top_p = kwargs.pop("top_p", None)
            # Common harness metadata that does not alter native decoding.
            kwargs.pop("max_length", None)
            if kwargs:
                LOGGER.warning("Ignoring unsupported generation kwargs: %s", sorted(kwargs))
            text = self.generate_text(
                context,
                max_new_tokens=max_new_tokens,
                until=stop,
                temperature=temperature,
                top_k=int(top_k) if top_k is not None else None,
                top_p=float(top_p) if top_p is not None else None,
                do_sample=do_sample,
            )
            results.append(text)
            self.cache_hook.add_partial("generate_until", request.args, text)
        return results
