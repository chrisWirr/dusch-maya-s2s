from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch

from maya_s2s.conversation import ConversationState
from maya_s2s.config import Settings, get_settings
from maya_s2s.llm import build_reply, clamp_reply
from maya_s2s.models import get_csm_stack, get_whisper_model


@dataclass(slots=True)
class S2SResult:
    transcript: str
    reply_text: str
    audio_path: str
    sample_rate: int


def transcribe_audio(input_path: Path) -> str:
    whisper = get_whisper_model()
    segments, _ = whisper.transcribe(str(input_path), vad_filter=True)
    return " ".join(segment.text.strip() for segment in segments).strip()


def synthesize_with_csm(text: str, speaker_id: int, output_path: Path) -> int:
    settings = get_settings()
    processor, model = get_csm_stack()
    conversation = [
        {"role": str(speaker_id), "content": [{"type": "text", "text": text}]},
    ]
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        return_dict=True,
    ).to(model.device)
    with torch.inference_mode():
        audio = model.generate(
            **inputs,
            output_audio=True,
            do_sample=settings.tts_do_sample,
            temperature=settings.tts_temperature,
            depth_decoder_do_sample=settings.tts_depth_decoder_do_sample,
            depth_decoder_temperature=settings.tts_depth_decoder_temperature,
        )
    processor.save_audio(audio, str(output_path))
    sample_rate = getattr(processor.feature_extractor, "sampling_rate", 24_000)
    append_tail_silence(output_path, sample_rate, settings.tts_tail_silence_ms)
    return sample_rate


def append_tail_silence(audio_path: Path, sample_rate: int, silence_ms: int) -> None:
    if silence_ms <= 0 or sample_rate <= 0:
        return
    data, current_sample_rate = sf.read(audio_path)
    target_sample_rate = int(current_sample_rate or sample_rate)
    silence_frames = int(target_sample_rate * silence_ms / 1000)
    if silence_frames <= 0:
        return
    if getattr(data, "ndim", 1) == 1:
        pad = np.zeros(silence_frames, dtype=data.dtype)
    else:
        pad = np.zeros((silence_frames, data.shape[1]), dtype=data.dtype)
    sf.write(audio_path, np.concatenate([data, pad], axis=0), target_sample_rate)


def run_pipeline(
    audio_path: Path,
    settings: Settings,
    system_prompt: str | None = None,
    speaker_id: int = 0,
    target_text: str | None = None,
    conversation: ConversationState | None = None,
) -> S2SResult:
    transcript = transcribe_audio(audio_path)
    reply_text = (
        target_text.strip()
        if target_text
        else build_reply(transcript, system_prompt, settings, conversation=conversation)
    )
    reply_text = clamp_reply(reply_text, settings)
    if conversation is not None:
        conversation.append_user(transcript)
        conversation.append_assistant(reply_text)
    output_path = settings.output_dir / f"reply-{uuid4().hex[:8]}.wav"
    sample_rate = synthesize_with_csm(reply_text, speaker_id=speaker_id, output_path=output_path)
    return S2SResult(
        transcript=transcript,
        reply_text=reply_text,
        audio_path=str(output_path),
        sample_rate=sample_rate,
    )


def pcm16_bytes_to_wav(raw_audio: bytes, destination: Path, sample_rate: int, channels: int) -> Path:
    frame_count = len(raw_audio) // 2
    if frame_count == 0:
        raise ValueError("No audio frames supplied.")

    audio = np.frombuffer(raw_audio, dtype="<i2")
    if channels > 1:
        audio = audio.reshape(-1, channels)
    audio = audio.astype("float32") / 32768.0
    sf.write(destination, audio, sample_rate, subtype="PCM_16")
    return destination


def normalize_uploaded_audio_bytes(content: bytes, destination: Path) -> Path:
    destination.write_bytes(content)
    try:
        data, sample_rate = sf.read(destination)
        sf.write(destination, data, sample_rate)
    except Exception:
        pass
    return destination
