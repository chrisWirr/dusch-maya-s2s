from __future__ import annotations

import re

from maya_s2s.conversation import ConversationState
from maya_s2s.config import Settings
from maya_s2s.models import get_text_model


def clamp_reply(text: str, settings: Settings) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= settings.max_tts_chars:
        return cleaned
    clipped = cleaned[: settings.max_tts_chars].rstrip()
    last_stop = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    if last_stop >= 40:
        return clipped[: last_stop + 1].strip()
    return f"{clipped.rstrip(' ,;:') }."


def build_reply(
    transcript: str,
    system_prompt: str | None,
    settings: Settings,
    conversation: ConversationState | None = None,
) -> str:
    prompt = (system_prompt or "Antworte hilfreich und knapp auf Deutsch.").strip()

    if settings.text_model_backend == "echo":
        if settings.echo_reply_style == "repeat":
            reply = transcript
        else:
            reply = "Verstanden."
        return clamp_reply(reply, settings)

    if settings.text_model_backend != "local":
        raise ValueError(
            f"Unsupported TEXT_MODEL_BACKEND={settings.text_model_backend!r}. "
            "Use 'echo' or 'local'."
        )

    tokenizer, model = get_text_model()
    if tokenizer is None or model is None:
        raise RuntimeError("Local text model could not be initialized.")

    if conversation is not None:
        if not conversation.system_prompt:
            conversation.system_prompt = prompt
        messages = conversation.as_messages(transcript, settings.history_turns)
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": transcript},
        ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        do_sample=settings.temperature > 0,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return clamp_reply(tokenizer.decode(new_tokens, skip_special_tokens=True), settings)
