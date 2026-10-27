"""Run a small open-weights code model locally with PyTorch + transformers."""

from __future__ import annotations

import os
from typing import Sequence

from .base import Completion, Message, Usage, UsageMeter


def pick_device() -> str:
    import torch

    forced = os.environ.get("MF_HF_DEVICE")
    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HFLocalLLM:
    is_simulation = False

    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf_local:{model_id}"
        self.meter = UsageMeter()
        self.device = pick_device()
        dtype = {"cuda": torch.bfloat16, "mps": torch.float16}.get(self.device, torch.float32)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(self.device)
        self.model.eval()

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> Completion:
        import torch

        inputs = self.tok.apply_chat_template(
            list(messages), add_generation_prompt=True, return_tensors="pt"
        ).to(self.device)
        gen_kwargs: dict = {"max_new_tokens": max_tokens, "pad_token_id": self.tok.eos_token_id}
        if temperature > 0:
            torch.manual_seed(0 if seed is None else seed)
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(inputs, **gen_kwargs)
        new_tokens = out[0, inputs.shape[-1] :]
        text = self.tok.decode(new_tokens, skip_special_tokens=True)
        usage = Usage(prompt_tokens=int(inputs.shape[-1]), completion_tokens=int(new_tokens.shape[-1]))
        self.meter.add(usage)
        return Completion(text=text, usage=usage, backend="hf_local")
