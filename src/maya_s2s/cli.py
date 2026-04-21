from __future__ import annotations

import argparse
import json
from pathlib import Path

from maya_s2s.config import get_settings
from maya_s2s.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Maya speech-to-speech pipeline.")
    parser.add_argument("--input", required=True, help="Path to an input audio file.")
    parser.add_argument("--prompt", default=None, help="Optional system prompt for the reply model.")
    parser.add_argument("--speaker-id", type=int, default=0, help="CSM speaker id.")
    parser.add_argument(
        "--target-text",
        default=None,
        help="If set, skip text generation and synthesize this text directly.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    result = run_pipeline(
        Path(args.input),
        settings=settings,
        system_prompt=args.prompt,
        speaker_id=args.speaker_id,
        target_text=args.target_text,
    )
    print(
        json.dumps(
            {
                "transcript": result.transcript,
                "reply_text": result.reply_text,
                "audio_path": result.audio_path,
                "sample_rate": result.sample_rate,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
