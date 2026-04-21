from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from maya_s2s.config import get_settings
from maya_s2s.pipeline import normalize_uploaded_audio_bytes, run_pipeline
from maya_s2s.streaming import (
    StreamingSession,
    decode_audio_chunk,
    new_session_id,
    parse_ws_message,
)

app = FastAPI(title="Maya Speech-to-Speech Pipeline", version="0.1.0")
SESSIONS: dict[str, StreamingSession] = {}
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/v1/speech-to-speech")
async def speech_to_speech(
    audio: UploadFile = File(...),
    prompt: str | None = Form(default=None),
    speaker_id: int = Form(default=0),
    target_text: str | None = Form(default=None),
):
    settings = get_settings()
    suffix = Path(audio.filename or "upload.wav").suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        tmp_path = Path(handle.name)
    normalize_uploaded_audio_bytes(await audio.read(), tmp_path)
    result = run_pipeline(
        tmp_path,
        settings=settings,
        system_prompt=prompt,
        speaker_id=speaker_id,
        target_text=target_text,
    )
    return {
        "transcript": result.transcript,
        "reply_text": result.reply_text,
        "audio_path": result.audio_path,
        "sample_rate": result.sample_rate,
    }


@app.websocket("/v1/ws/speech-to-speech")
async def speech_to_speech_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    session = StreamingSession(session_id=new_session_id(), settings=settings)
    SESSIONS[session.session_id] = session
    await websocket.send_json(
        {
            "type": "ready",
            "session_id": session.session_id,
            "sample_rate": settings.stream_sample_rate,
            "channels": settings.stream_channels,
            "format": "pcm_s16le",
        }
    )
    try:
        while True:
            message = await websocket.receive()

            if "text" in message and message["text"] is not None:
                payload = parse_ws_message(message["text"])
                msg_type = payload.get("type")

                if msg_type == "config":
                    session.update_config(
                        prompt=payload.get("prompt") if isinstance(payload.get("prompt"), str) else None,
                        speaker_id=int(payload["speaker_id"])
                        if payload.get("speaker_id") is not None
                        else None,
                    )
                    await websocket.send_json(
                        {
                            "type": "config_applied",
                            "session_id": session.session_id,
                            "speaker_id": session.conversation.speaker_id,
                        }
                    )
                    continue

                if msg_type == "audio":
                    chunk = decode_audio_chunk(payload)
                    for event in session.feed_audio(chunk):
                        await websocket.send_json(
                            {
                                "type": "vad",
                                "event": event.event,
                                "duration_ms": event.duration_ms,
                            }
                        )
                        if event.raw_audio:
                            await websocket.send_json(
                                session.process_turn(
                                    event.raw_audio,
                                    target_text=payload.get("target_text")
                                    if isinstance(payload.get("target_text"), str)
                                    else None,
                                )
                            )
                    continue

                if msg_type == "flush":
                    event = session.flush()
                    if event and event.raw_audio:
                        await websocket.send_json(
                            {
                                "type": "vad",
                                "event": event.event,
                                "duration_ms": event.duration_ms,
                            }
                        )
                        await websocket.send_json(
                            session.process_turn(
                                event.raw_audio,
                                target_text=payload.get("target_text")
                                if isinstance(payload.get("target_text"), str)
                                else None,
                            )
                        )
                    else:
                        await websocket.send_json({"type": "noop", "reason": "no_buffered_audio"})
                    continue

                if msg_type == "reset":
                    session.reset_all()
                    await websocket.send_json({"type": "reset_done", "session_id": session.session_id})
                    continue

                await websocket.send_json({"type": "error", "error": f"Unsupported message type: {msg_type}"})
                continue

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                for event in session.feed_audio(chunk):
                    await websocket.send_json(
                        {
                            "type": "vad",
                            "event": event.event,
                            "duration_ms": event.duration_ms,
                        }
                    )
                    if event.raw_audio:
                        await websocket.send_json(session.process_turn(event.raw_audio))
                continue

    except WebSocketDisconnect:
        SESSIONS.pop(session.session_id, None)
        return
    except Exception as exc:
        SESSIONS.pop(session.session_id, None)
        await websocket.send_json({"type": "error", "error": str(exc)})
        await websocket.close(code=1011)
        return


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "maya_s2s.server:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
