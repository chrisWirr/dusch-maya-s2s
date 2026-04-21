const els = {
  prompt: document.getElementById("prompt"),
  speakerId: document.getElementById("speakerId"),
  wsUrl: document.getElementById("wsUrl"),
  connectBtn: document.getElementById("connectBtn"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  flushBtn: document.getElementById("flushBtn"),
  resetBtn: document.getElementById("resetBtn"),
  status: document.getElementById("status"),
  sessionId: document.getElementById("sessionId"),
  streamState: document.getElementById("streamState"),
  vadState: document.getElementById("vadState"),
  messages: document.getElementById("messages"),
  replyAudio: document.getElementById("replyAudio"),
};

const state = {
  socket: null,
  audioContext: null,
  mediaStream: null,
  sourceNode: null,
  processorNode: null,
  isStreaming: false,
};

function defaultWsUrl() {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/v1/ws/speech-to-speech`;
}

function setStatus(text) {
  els.status.textContent = text;
}

function setStreamState(text) {
  els.streamState.textContent = text;
}

function setVad(text) {
  els.vadState.textContent = text;
}

function addMessage(role, text) {
  const node = document.createElement("article");
  node.className = `message ${role}`;
  node.textContent = text;
  els.messages.prepend(node);
}

function updateControls() {
  const connected = state.socket && state.socket.readyState === WebSocket.OPEN;
  els.connectBtn.disabled = connected;
  els.startBtn.disabled = !connected || state.isStreaming;
  els.stopBtn.disabled = !connected || !state.isStreaming;
  els.flushBtn.disabled = !connected;
  els.resetBtn.disabled = !connected;
}

function floatTo16BitPCM(float32Array) {
  const pcm = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, float32Array[i]));
    pcm[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return pcm.buffer;
}

async function connect() {
  const url = els.wsUrl.value.trim();
  state.socket = new WebSocket(url);
  state.socket.binaryType = "arraybuffer";

  state.socket.onopen = () => {
    setStatus("Verbunden");
    updateControls();
    state.socket.send(
      JSON.stringify({
        type: "config",
        prompt: els.prompt.value,
        speaker_id: Number(els.speakerId.value || 0),
      })
    );
  };

  state.socket.onclose = () => {
    setStatus("Getrennt");
    els.sessionId.textContent = "-";
    setStreamState("Aus");
    setVad("Idle");
    stopStreaming();
    state.socket = null;
    updateControls();
  };

  state.socket.onerror = () => {
    setStatus("Fehler");
  };

  state.socket.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    if (payload.type === "ready") {
      els.sessionId.textContent = payload.session_id;
      setStatus("Bereit");
      addMessage("system", `Session ${payload.session_id} bereit (${payload.sample_rate} Hz).`);
      return;
    }
    if (payload.type === "config_applied") {
      addMessage("system", `Konfiguration gesetzt. Speaker ID ${payload.speaker_id}.`);
      return;
    }
    if (payload.type === "vad") {
      setVad(`${payload.event} (${payload.duration_ms} ms)`);
      return;
    }
    if (payload.type === "turn_result") {
      addMessage("user", payload.transcript);
      addMessage("assistant", payload.reply_text);
      const audioUrl = `data:audio/wav;base64,${payload.audio_base64}`;
      els.replyAudio.src = audioUrl;
      els.replyAudio.play().catch(() => {});
      setVad("Idle");
      return;
    }
    if (payload.type === "reset_done") {
      addMessage("system", "Dialogzustand zurückgesetzt.");
      return;
    }
    if (payload.type === "noop") {
      addMessage("system", payload.reason);
      return;
    }
    if (payload.type === "error") {
      addMessage("system", `Fehler: ${payload.error}`);
    }
  };
}

async function startStreaming() {
  if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
    return;
  }
  state.audioContext = new AudioContext({ sampleRate: 16000 });
  state.mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });
  state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
  state.processorNode = state.audioContext.createScriptProcessor(4096, 1, 1);
  state.processorNode.onaudioprocess = (event) => {
    if (!state.socket || state.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    const channel = event.inputBuffer.getChannelData(0);
    const pcm = floatTo16BitPCM(channel);
    state.socket.send(pcm);
  };
  state.sourceNode.connect(state.processorNode);
  state.processorNode.connect(state.audioContext.destination);
  state.isStreaming = true;
  setStreamState("An");
  updateControls();
}

function stopStreaming() {
  if (state.processorNode) {
    state.processorNode.disconnect();
    state.processorNode.onaudioprocess = null;
    state.processorNode = null;
  }
  if (state.sourceNode) {
    state.sourceNode.disconnect();
    state.sourceNode = null;
  }
  if (state.mediaStream) {
    state.mediaStream.getTracks().forEach((track) => track.stop());
    state.mediaStream = null;
  }
  if (state.audioContext) {
    state.audioContext.close().catch(() => {});
    state.audioContext = null;
  }
  state.isStreaming = false;
  setStreamState("Aus");
  updateControls();
}

function flushTurn() {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ type: "flush" }));
  }
}

function resetConversation() {
  if (state.socket && state.socket.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify({ type: "reset" }));
  }
}

els.wsUrl.value = defaultWsUrl();
els.connectBtn.addEventListener("click", connect);
els.startBtn.addEventListener("click", startStreaming);
els.stopBtn.addEventListener("click", stopStreaming);
els.flushBtn.addEventListener("click", flushTurn);
els.resetBtn.addEventListener("click", resetConversation);
updateControls();
