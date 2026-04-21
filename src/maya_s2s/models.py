from __future__ import annotations

import logging
from functools import lru_cache

import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, CsmForConditionalGeneration

from maya_s2s.config import Settings, get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def cuda_runtime_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        # Force a lightweight runtime touch so we don't trust a broken CUDA setup.
        torch.zeros(1, device="cuda")
        return True
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        logger.warning("CUDA runtime unavailable, falling back to CPU: %s", exc)
        return False


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if cuda_runtime_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str):
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float32":
        return torch.float32
    if device == "cuda":
        return torch.float16
    return torch.float32


@lru_cache(maxsize=1)
def get_runtime_settings() -> Settings:
    return get_settings()


@lru_cache(maxsize=1)
def get_whisper_model() -> WhisperModel:
    settings = get_runtime_settings()
    device = resolve_device(settings.device)
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(settings.whisper_model_size, device=device, compute_type=compute_type)
    except RuntimeError as exc:
        if device != "cuda":
            raise
        logger.warning("Falling back to CPU Whisper after CUDA init failure: %s", exc)
        return WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def get_text_model():
    settings = get_runtime_settings()
    if settings.text_model_backend != "local":
        return None, None

    requested_device = resolve_device(settings.device)
    if requested_device == "cpu":
        raise RuntimeError("Local text model requires an accelerated device for acceptable latency.")
    tokenizer = AutoTokenizer.from_pretrained(settings.text_model_id)

    dtype = resolve_dtype(settings.torch_dtype, requested_device)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            settings.text_model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        model.to(requested_device)
        return tokenizer, model
    except RuntimeError as exc:
        logger.warning("Local text model unavailable on %s: %s", requested_device, exc)
        raise


@lru_cache(maxsize=1)
def get_csm_stack():
    settings = get_runtime_settings()
    processor = AutoProcessor.from_pretrained(settings.csm_model_id)
    requested_device = resolve_device(settings.device)

    for device in [requested_device, "cpu"]:
        if device == "cpu" and requested_device == "cpu":
            continue
        dtype = resolve_dtype(settings.torch_dtype, device)
        try:
            model = CsmForConditionalGeneration.from_pretrained(
                settings.csm_model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            return processor, model
        except RuntimeError as exc:
            if device != "cuda":
                raise
            logger.warning("Falling back to CPU CSM after CUDA init failure: %s", exc)

    model = CsmForConditionalGeneration.from_pretrained(
        settings.csm_model_id,
        torch_dtype=resolve_dtype(settings.torch_dtype, "cpu"),
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    return processor, model
