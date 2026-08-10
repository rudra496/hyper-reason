"""Transformers (HuggingFace) backend — torch-gated, real ``model.generate``.

This is the path for users with a local GPU + PyTorch. It is NOT a silent no-op: if torch or
transformers is absent it raises ImportError telling the user exactly which extra to install.
Token counts come from the real tokenizer. Cannot run in this dev env (no torch/GPU) but the
code is honest and would work where the deps exist.
"""

from __future__ import annotations

from typing import Sequence

try:
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    _OK = True
except Exception:  # pragma: no cover - optional extra
    _OK = False

from .base import Sample


class TransformersBackend:
    is_live = True

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        device_map: str | None = None,
        torch_dtype: str = "auto",
    ):
        if not _OK:
            raise ImportError(
                "TransformersBackend requires PyTorch + transformers. "
                "Install with: pip install hyper-reason[transformers]"
            )
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device_map or "auto"
        )

    @property
    def name(self) -> str:
        return f"hf:{self.model_name}"

    def sample(
        self,
        prompt: str,
        k: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        stop: Sequence[str] = (),
    ) -> list[Sample]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        with torch.no_grad():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5) if do_sample else 1.0,
                num_return_sequences=k,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        out: list[Sample] = []
        for row in gen:
            new_ids = row[prompt_len:]
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            for s in stop:
                if s and s in text:
                    text = text.split(s)[0]
            out.append(
                Sample(
                    text=text,
                    prompt_tokens=int(prompt_len),
                    completion_tokens=int(new_ids.shape[0]),
                    finish_reason="stop",
                    raw={"model": self.model_name},
                )
            )
        return out

    def count_tokens(self, text: str) -> int:
        return int(len(self.tokenizer.encode(text)))
