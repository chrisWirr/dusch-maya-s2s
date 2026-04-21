from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import numpy as np

from maya_s2s.config import Settings
from maya_s2s.conversation import ConversationState
from maya_s2s.pipeline import pcm16_bytes_to_wav, run_pipeline


@dataclass(slots=True)
class TurnDetectionResult:
    event: str
    duration_ms: int
    raw_audio: bytes | None = None


@dataclass(slots=True)
class StreamingSession:
    session_id: str
    settings: Settings
    conversation: ConversationState = field(default_factory=ConversationState)
    audio_buffer: bytearray = field(default_factory=bytearray)
    speech_started: bool = False
    speech_ms: int = 0
    silence_ms: int = 0

    def update_config(self, prompt: str | None = None, speaker_id: int | None = None) -> None:
        if prompt is not None:
            self.conversation.system_prompt = prompt
        if speaker_id is not None:
            self.conversation.speaker_id = speaker_id

    def reset_audio_state(self) -> None:
        self.audio_buffer.clear()
        self.speech_started = False
        self.speech_ms = 0
        self.silence_ms = 0

    def reset_all(self) -> None:
        self.reset_audio_state()
        self.conversation.turns.clear()

    def feed_audio(self, chunk: bytes) -> list[TurnDetectionResult]:
        if not chunk:
            return []
        duration_ms = chunk_duration_ms(
            len(chunk), self.settings.stream_sample_rate, self.settings.stream_channels
        )
        rms = pcm16_rms(chunk)
        events: list[TurnDetectionResult] = []

        if rms >= self.settings.vad_rms_threshold:
            if not self.speech_started:
                self.speech_started = True
                events.append(TurnDetectionResult(event="speech_start", duration_ms=duration_ms))
            self.audio_buffer.extend(chunk)
            self.speech_ms += duration_ms
            self.silence_ms = 0
        elif self.speech_started:
            self.audio_buffer.extend(chunk)
            self.silence_ms += duration_ms

        if self.speech_started and self.speech_ms >= self.settings.vad_min_speech_ms:
            if self.silence_ms >= self.settings.vad_end_silence_ms:
                raw_audio = bytes(self.audio_buffer)
                total_ms = self.speech_ms + self.silence_ms
                self.reset_audio_state()
                events.append(
                    TurnDetectionResult(
                        event="speech_end",
                        duration_ms=total_ms,
                        raw_audio=raw_audio,
                    )
                )
                return events

        if self.speech_started and (self.speech_ms + self.silence_ms) >= self.settings.vad_max_turn_ms:
            raw_audio = bytes(self.audio_buffer)
            total_ms = self.speech_ms + self.silence_ms
            self.reset_audio_state()
            events.append(
                TurnDetectionResult(
                    event="max_turn_reached",
                    duration_ms=total_ms,
                    raw_audio=raw_audio,
                )
            )
        return events

    def flush(self) -> TurnDetectionResult | None:
        if not self.audio_buffer:
            return None
        raw_audio = bytes(self.audio_buffer)
        total_ms = self.speech_ms + self.silence_ms
        self.reset_audio_state()
        return TurnDetectionResult(event="flush", duration_ms=total_ms, raw_audio=raw_audio)

    def process_turn(self, raw_audio: bytes, target_text: str | None = None) -> dict[str, object]:
        with NamedTemporaryFile(delete=False, suffix=".wav") as handle:
            input_path = Path(handle.name)
        pcm16_bytes_to_wav(
            raw_audio,
            input_path,
            sample_rate=self.settings.stream_sample_rate,
            channels=self.settings.stream_channels,
        )
        result = run_pipeline(
            input_path,
            settings=self.settings,
            system_prompt=self.conversation.system_prompt,
            speaker_id=self.conversation.speaker_id,
            target_text=target_text,
            conversation=self.conversation,
        )
        output_bytes = Path(result.audio_path).read_bytes()
        return {
            "type": "turn_result",
            "transcript": result.transcript,
            "reply_text": result.reply_text,
            "audio_path": result.audio_path,
            "sample_rate": result.sample_rate,
            "audio_base64": base64.b64encode(output_bytes).decode("ascii"),
        }


def chunk_duration_ms(byte_length: int, sample_rate: int, channels: int) -> int:
    if sample_rate <= 0 or channels <= 0:
        return 0
    samples = byte_length / 2 / channels
    return int(samples / sample_rate * 1000)


def pcm16_rms(chunk: bytes) -> float:
    if not chunk:
        return 0.0
    audio = np.frombuffer(chunk, dtype="<i2").astype("float32") / 32768.0
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def parse_ws_message(text: str) -> dict[str, object]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("WebSocket message must be a JSON object.")
    return payload


def decode_audio_chunk(payload: dict[str, object]) -> bytes:
    encoded = payload.get("audio")
    if not isinstance(encoded, str):
        raise ValueError("Audio payload must be base64 encoded.")
    return base64.b64decode(encoded)


def new_session_id() -> str:
    return uuid4().hex[:12]
