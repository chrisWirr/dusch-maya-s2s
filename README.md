# Maya Speech-to-Speech Pipeline

Diese Pipeline baut eine lokale oder cloudfähige Speech-to-Speech-Anwendung um `sesame/csm-1b`. Wichtig ist die Modellgrenze:

- Öffentlich verfügbar ist `sesame/csm-1b`.
- Sesame schrieb am 13. März 2025, dass eine feinabgestimmte Variante davon `Maya` antreibt.
- Diese proprietäre `Maya`-Finetune ist nicht öffentlich. Diese Pipeline nutzt daher das gleiche öffentliche `CSM`-Basismodell, nicht den privaten exakten Maya-Checkpoint.

## Architektur

1. `faster-whisper` transkribiert das Eingangsaudio.
2. Ein Textmodell erzeugt die Antwort.
3. `sesame/csm-1b` erzeugt die Sprachausgabe.
4. Optionaler WebSocket-Modus übernimmt Chunking, Turn-Detection und Dialogzustand.

Die Textschicht ist bewusst austauschbar:

- `local` nutzt ein lokales Hugging-Face-Causal-LM.
- `echo` ist ein Debug-Modus mit bewusst sehr kurzer Standardantwort.

## Anforderungen

- NVIDIA-GPU empfohlen, lokal oder in der Cloud.
- Python 3.10+ oder Docker mit NVIDIA Container Toolkit.
- Hugging Face Zugriff auf:
  - `sesame/csm-1b`

Hinweis: Für sinnvolle Latenz ist GPU praktisch Pflicht. CPU ist nur als langsamer Fallback vorgesehen.

## Schnellstart lokal

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
copy .env.example .env
```

Dann `.env` anpassen, insbesondere:

```env
CSM_MODEL_ID=sesame/csm-1b
TEXT_MODEL_BACKEND=echo
TEXT_MODEL_ID=Qwen/Qwen2.5-3B-Instruct
WHISPER_MODEL_SIZE=tiny
DEVICE=auto
TORCH_DTYPE=auto
OUTPUT_DIR=artifacts
MAX_TTS_CHARS=180
ECHO_REPLY_STYLE=brief
```

API starten:

```powershell
.\.venv\Scripts\maya-s2s-server.exe
```

CLI nutzen:

```powershell
.\.venv\Scripts\maya-s2s.exe --input .\input.wav --prompt "Antworte kurz und freundlich auf Deutsch."
```

Browser-Demo:

1. Server starten
2. [http://127.0.0.1:8000](http://127.0.0.1:8000) im Browser öffnen
3. Verbinden und dann Aufnahme starten

## Docker / lokale GPU

Build:

```powershell
docker build -t maya-s2s .
```

Run:

```powershell
docker run --gpus all --rm -p 8000:8000 --env-file .env maya-s2s
```

## Cloud-GPU

Die gleiche Container-Image-Strategie funktioniert auf GPU-Anbietern wie Runpod, Vast.ai, Lambda oder Hugging Face Docker-Workloads.

Wichtig:

- dieselbe `.env` verwenden
- persistentes Volume für `artifacts/` und Hugging Face Cache mounten
- `HF_TOKEN` als Secret setzen, falls der Runtime-Kontext Auth braucht

Beispiel-Startkommando im Container:

```bash
uvicorn maya_s2s.server:app --host 0.0.0.0 --port 8000
```

## Lightning AI

Fuer `lightning.ai` ist dieses Repo jetzt direkt als GPU-Service nutzbar. Der Server liest dafuer:

- `APP_HOST`, Standard `0.0.0.0`
- `APP_PORT`, Standard `8000`
- `PORT`, falls Lightning den Port dynamisch injiziert; `PORT` hat Vorrang vor `APP_PORT`

Empfohlener Ablauf in einem Lightning Studio oder Work:

1. Repo auf Lightning klonen oder hochladen
2. Python 3.10 bis 3.12 mit GPU-Runner waehlen
3. Abhaengigkeiten installieren:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

4. `.env` setzen, mindestens:

```env
CSM_MODEL_ID=sesame/csm-1b
TEXT_MODEL_BACKEND=echo
WHISPER_MODEL_SIZE=tiny
DEVICE=cuda
TORCH_DTYPE=float16
HF_HOME=/teamspace/studios/this_studio/.cache/huggingface
OUTPUT_DIR=/teamspace/studios/this_studio/artifacts
```

5. Server starten:

```bash
python -m maya_s2s.server
```

Wenn du den Dienst als oeffentliche Lightning-App betreibst, binde den von Lightning gesetzten `PORT` durchgereicht ein. Das Repo unterstuetzt das jetzt ohne weiteren Code.

### Studio setup in this repo

Dieses Repo bringt jetzt eine einfache Lightning-Studio-Struktur mit:

- `.lightning_studio/on_start.sh` bootstrappt beim Studio-Start die lokale `.venv`, installiert das Projekt und legt Cache- sowie Artifact-Ordner an.
- `scripts/run_lightning_server.sh` startet den FastAPI-Dienst mit Lightning-tauglichen Defaults fuer `PORT`, `HF_HOME`, `OUTPUT_DIR`, `DEVICE` und `TORCH_DTYPE`.
- `requirements.txt` zeigt Studio-Tools einen minimalen Python-Installpfad.

Empfohlener Ablauf im Lightning-UI:

1. Neues Studio erstellen
2. Dieses Projekt in `/teamspace/studios/this_studio` laden oder klonen
3. Sicherstellen, dass eine GPU-Maschine aktiv ist
4. Optional `.env` anpassen
5. Im Studio-Terminal ausfuehren:

```bash
bash .lightning_studio/on_start.sh
bash scripts/run_lightning_server.sh
```

Danach sollte die App im Studio auf dem von Lightning bereitgestellten Port erreichbar sein.

## API

`POST /v1/speech-to-speech`

Multipart-Form:

- `audio`: WAV/MP3/FLAC
- `prompt`: optionaler Systemprompt
- `speaker_id`: optional, Standard `0`
- `target_text`: optional; wenn gesetzt, wird keine LLM-Antwort generiert, sondern dieser Text direkt vertont

Antwort:

```json
{
  "transcript": "Hallo zusammen",
  "reply_text": "Hallo, wie kann ich helfen?",
  "audio_path": "artifacts/reply.wav",
  "sample_rate": 24000
}
```

## Streaming / Realtime

`WS /v1/ws/speech-to-speech`

Der WebSocket erwartet standardmäßig `PCM S16LE`, mono, `16000 Hz`. Nach Connect kommt:

```json
{
  "type": "ready",
  "session_id": "abc123",
  "sample_rate": 16000,
  "channels": 1,
  "format": "pcm_s16le"
}
```

Danach typischer Ablauf:

1. optional `config` senden
2. Audio-Chunks senden, entweder als Binary-Frames oder als JSON mit Base64
3. Server erkennt Turn-Ende über einfache RMS-VAD
4. Server antwortet mit `vad` und anschließend `turn_result`

Beispiel `config`:

```json
{
  "type": "config",
  "prompt": "Antworte warm, direkt und auf Deutsch.",
  "speaker_id": 0
}
```

Beispiel `audio` als JSON:

```json
{
  "type": "audio",
  "audio": "<base64 pcm bytes>"
}
```

Flush eines angefangenen Turns:

```json
{
  "type": "flush"
}
```

Antwort bei fertigem Turn:

```json
{
  "type": "turn_result",
  "transcript": "Kannst du mir helfen?",
  "reply_text": "Ja. Wobei genau brauchst du Hilfe?",
  "audio_path": "artifacts/reply-1234abcd.wav",
  "sample_rate": 24000,
  "audio_base64": "<base64 wav>"
}
```

## Turn-Detection

Die Realtime-Variante nutzt hier absichtlich eine simple, robuste serverseitige Turn-Detection:

- RMS-basierte Sprachaktivität
- Mindestsprachdauer
- End-of-turn über Stillefenster
- maximale Turn-Länge als Schutz

Die Parameter sind über `.env` steuerbar:

```env
STREAM_SAMPLE_RATE=16000
STREAM_CHANNELS=1
VAD_RMS_THRESHOLD=0.015
VAD_MIN_SPEECH_MS=280
VAD_END_SILENCE_MS=700
VAD_MAX_TURN_MS=12000
HISTORY_TURNS=6
```

## Modellentscheidung

`sesame/csm-1b` ist hier absichtlich über `transformers` integriert statt über das ursprüngliche Sesame-Repo:

- weniger proprietäre Hilfslogik im App-Code
- einfacher Container-Deploy
- gleiche öffentliche Modell-ID

## Nächste sinnvolle Ausbaustufen

- silero-vad oder serverseitige Neural-VAD statt RMS-Heuristik
- inkrementelles STT und partielle Hypothesen
- echter Dialogspeicher mit Audio-Kontextsegmenten
- optionaler OpenAI-kompatibler Text-Backend-Adapter
