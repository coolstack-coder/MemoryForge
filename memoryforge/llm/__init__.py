from __future__ import annotations

from .base import LLM, Completion, LLMError, Message, Usage, UsageMeter
from .simulated import SimConfig, SimulatedLLM


def make_llm(backend: str | None = None) -> LLM:
    """Build the backend named by `backend` or MF_LLM_BACKEND."""
    from ..config import settings

    backend = backend or settings.llm_backend
    if backend == "simulated":
        return SimulatedLLM()
    if backend == "openai_compatible":
        from .openai_compat import OpenAICompatibleLLM

        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    if backend == "hf_local":
        from .hf_local import HFLocalLLM

        return HFLocalLLM(settings.hf_model)
    raise ValueError(f"unknown LLM backend {backend!r}")


__all__ = ["LLM", "Completion", "LLMError", "Message", "SimConfig", "SimulatedLLM", "Usage", "UsageMeter", "make_llm"]
