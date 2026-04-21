from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    csm_model_id: str = os.getenv("CSM_MODEL_ID", "sesame/csm-1b")
    text_model_backend: str = os.getenv("TEXT_MODEL_BACKEND", "echo")
    text_model_id: str = os.getenv("TEXT_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    device: str = os.getenv("DEVICE", "auto")
    torch_dtype: str = os.getenv("TORCH_DTYPE", "auto")
    tts_do_sample: bool = os.getenv("TTS_DO_SAMPLE", "false").lower() == "true"
    tts_temperature: float = float(os.getenv("TTS_TEMPERATURE", "1.0"))
    tts_depth_decoder_do_sample: bool = (
        os.getenv("TTS_DEPTH_DECODER_DO_SAMPLE", "false").lower() == "true"
    )
    tts_depth_decoder_temperature: float = float(
        os.getenv("TTS_DEPTH_DECODER_TEMPERATURE", "1.0")
    )
    tts_tail_silence_ms: int = int(os.getenv("TTS_TAIL_SILENCE_MS", "180"))
    app_host: str = os.getenv("APP_HOST", "0.0.0.0")
    app_port: int = int(os.getenv("PORT", os.getenv("APP_PORT", "8000")))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", "artifacts"))
    max_new_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "192"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    history_turns: int = int(os.getenv("HISTORY_TURNS", "6"))
    max_tts_chars: int = int(os.getenv("MAX_TTS_CHARS", "180"))
    echo_reply_style: str = os.getenv("ECHO_REPLY_STYLE", "brief")
    stream_sample_rate: int = int(os.getenv("STREAM_SAMPLE_RATE", "16000"))
    stream_channels: int = int(os.getenv("STREAM_CHANNELS", "1"))
    vad_rms_threshold: float = float(os.getenv("VAD_RMS_THRESHOLD", "0.015"))
    vad_min_speech_ms: int = int(os.getenv("VAD_MIN_SPEECH_MS", "280"))
    vad_end_silence_ms: int = int(os.getenv("VAD_END_SILENCE_MS", "700"))
    vad_max_turn_ms: int = int(os.getenv("VAD_MAX_TURN_MS", "12000"))


def get_settings() -> Settings:
    settings = Settings()
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    return settings
