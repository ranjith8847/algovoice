/* Real-time voice interviewer browser client.
 *
 * Responsibilities:
 * - Capture microphone audio continuously (16 kHz mono PCM16 via WebSocket)
 * - Display the conversation as it happens (states, transcripts, replies)
 * - Play the interviewer's sentences in order, and stop immediately on
 *   interruption (barge-in)
 */

const SAMPLE_RATE = 16000;   // must match the backend pipeline
const CHUNK_SAMPLES = 960;   // 60 ms of audio per WebSocket message

const STATE_LABELS = {
  idle: "Idle",
  listening: "Listening…",
  processing: "Transcribing…",
  thinking: "Thinking…",
  speaking: "Interviewer speaking…",
  finishing: "Wrapping up…",
  finished: "Finished",
};

const els = {
  statusChip: document.getElementById("statusChip"),
  timer: document.getElementById("timer"),
  questionSelect: document.getElementById("questionSelect"),
  questionInfo: document.getElementById("questionInfo"),
  problemStatement: document.getElementById("problemStatement"),
  durationSelect: document.getElementById("durationSelect"),
  nameInput: document.getElementById("nameInput"),
  startButton: document.getElementById("startButton"),
  interruptButton: document.getElementById("interruptButton"),
  endButton: document.getElementById("endButton"),
  resetButton: document.getElementById("resetButton"),
  chat: document.getElementById("chat"),
  micNotice: document.getElementById("micNotice"),
  statePanel: document.getElementById("statePanel"),
  spPhase: document.getElementById("spPhase"),
  spHints: document.getElementById("spHints"),
  spAction: document.getElementById("spAction"),
  spComplexity: document.getElementById("spComplexity"),
  spIssues: document.getElementById("spIssues"),
};

let ws = null;
let audioContext = null;
let mediaStream = null;
let micProcessor = null;
let micSource = null;
let leftover = new Float32Array(0);
let startedAt = null;
let timerInterval = null;
let activeSources = [];
let nextPlayTime = 0;
let assistantBubble = null;

// Playback turn bookkeeping: the server tags a generation on each audio
// turn; playback_idle echoes it so stale reports are rejected server-side.
let currentTurnGen = null;
let answerEndClientTs = null;   // client timestamp of your last answer
let firstAudibleMs = null;      // first audible audio after that answer

// --------------------------------------------------------------------------
// Setup panel
// --------------------------------------------------------------------------
async function populateQuestions() {
  try {
    const response = await fetch("/questions");
    const bank = await response.json();
    Object.entries(bank).forEach(([id, q]) => {
      const option = document.createElement("option");
      option.value = id;
      option.textContent = id;
      option.dataset.topic = q.topic;
      option.dataset.difficulty = q.difficulty;
      option.dataset.statement = q.statement;
      els.questionSelect.appendChild(option);
    });
    updateQuestionInfo();
  } catch {
    addToast("error", "Could not load the question bank.");
  }
}

function updateQuestionInfo() {
  const option = els.questionSelect.selectedOptions[0];
  if (!option) return;
  els.questionInfo.innerHTML =
    `<b>Topic:</b> ${option.dataset.topic} &nbsp; ` +
    `<b>Difficulty:</b> ${option.dataset.difficulty}`;
  els.problemStatement.textContent = option.dataset.statement || "";
}

els.questionSelect.addEventListener("change", updateQuestionInfo);

// --------------------------------------------------------------------------
// WebSocket
// --------------------------------------------------------------------------
function connect() {
  const protocol = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${protocol}://${location.host}/ws`);
  ws.binaryType = "arraybuffer";
  ws.onclose = () => setStatus("idle");
  ws.onerror = () => addToast("error", "Connection to the server failed.");
  ws.onmessage = onServerMessage;
}

function sendEvent(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

function onServerMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    playAudioSentence(event.data);
    return;
  }

  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    return;
  }

  switch (msg.type) {
    case "state":
      setStatus(msg.state);
      els.interruptButton.disabled =
        !(msg.state === "listening" || msg.state === "speaking" ||
          msg.state === "thinking");
      if (msg.state === "finished") stopMic();
      break;

    case "interview_started":
      onInterviewStarted(msg);
      break;

    case "transcript":
      assistantBubble = null; // interviewer reply opens a fresh bubble
      answerEndClientTs = performance.now();
      addBubble("user", msg.text);
      break;

    case "assistant_delta":
      appendAssistantText(msg.text);
      break;

    case "metrics":
      if (assistantBubble && msg.first_audio_ms != null) {
        const parts = [];
        if (msg.llm_first_ms != null) parts.push(`first token ${msg.llm_first_ms} ms`);
        parts.push(`first audio ${msg.first_audio_ms} ms after your answer`);
        const meta = document.createElement("span");
        meta.className = "meta";
        meta.textContent = parts.join(" · ");
        assistantBubble.appendChild(meta);
      }
      break;

    case "stop_speaking":
      stopPlayback();
      break;

    case "turn_audio":
      // The server began streaming audio for this generation.
      currentTurnGen = msg.generation;
      firstAudibleMs = null;
      break;

    case "interrupted":
      assistantBubble = null;
      addNotice("You interrupted the interviewer — keep going.");
      break;

    case "interview_update":
      updateStatePanel(msg);
      break;

    case "error":
      addToast("error", msg.message);
      break;
  }
}

// --------------------------------------------------------------------------
// Mic capture
// --------------------------------------------------------------------------
async function startMic() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,  // keeps the interviewer's own voice out
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    if (audioContext.state === "suspended") await audioContext.resume();

    micSource = audioContext.createMediaStreamSource(mediaStream);
    micProcessor = audioContext.createScriptProcessor(4096, 1, 1);
    leftover = new Float32Array(0);

    micProcessor.onaudioprocess = (event) =>
      captureProcess(event.inputBuffer.getChannelData(0));

    micSource.connect(micProcessor);
    micProcessor.connect(audioContext.destination);
    els.micNotice.classList.add("hidden");
  } catch (err) {
    addToast("error", "Microphone access failed: " + err.message);
  }
}

function stopMic() {
  if (micProcessor) { micProcessor.disconnect(); micProcessor = null; }
  if (micSource) { micSource.disconnect(); micSource = null; }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function captureProcess(input) {
  let samples = input;
  if (audioContext.sampleRate !== SAMPLE_RATE) {
    samples = resample(input, audioContext.sampleRate, SAMPLE_RATE);
  }

  const combined = new Float32Array(leftover.length + samples.length);
  combined.set(leftover, 0);
  combined.set(samples, leftover.length);

  const usable = Math.floor(combined.length / CHUNK_SAMPLES) * CHUNK_SAMPLES;
  leftover = combined.slice(usable);

  for (let offset = 0; offset < usable; offset += CHUNK_SAMPLES) {
    sendPcmChunk(combined.slice(offset, offset + CHUNK_SAMPLES));
  }
}

function sendPcmChunk(chunk) {
  const pcm = new Int16Array(chunk.length);
  for (let i = 0; i < chunk.length; i++) {
    const s = Math.max(-1, Math.min(1, chunk[i]));
    pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(pcm.buffer);
  }
}

function resample(input, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const length = Math.floor(input.length / ratio);
  const out = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    const pos = i * ratio;
    const left = Math.floor(pos);
    const frac = pos - left;
    const a = input[left] || 0;
    const b = input[left + 1] || a;
    out[i] = a + (b - a) * frac;
  }
  return out;
}

// --------------------------------------------------------------------------
// Playback (sentence queue with interruption)
// --------------------------------------------------------------------------
function playAudioSentence(wavBytes) {
  if (!audioContext) return;
  audioContext.decodeAudioData(
    wavBytes.slice(0),
    (buffer) => {
      const source = audioContext.createBufferSource();
      source.buffer = buffer;
      source.connect(audioContext.destination);

      const startAt = Math.max(audioContext.currentTime + 0.02, nextPlayTime);
      source.start(startAt);
      nextPlayTime = startAt + buffer.duration;

      if (firstAudibleMs === null && answerEndClientTs !== null) {
        // Sound becomes audible at startAt; approximating with the
        // scheduling time (the offset to the AudioContext clock is small).
        firstAudibleMs = Math.max(
          0,
          Math.round(performance.now() - answerEndClientTs)
        );
      }

      activeSources.push(source);
      source.onended = () => {
        const index = activeSources.indexOf(source);
        if (index >= 0) activeSources.splice(index, 1);
        if (activeSources.length === 0) {
          nextPlayTime = 0;
          // Tell the server playback really finished so it leaves
          // "speaking" only when the user can no longer hear the
          // interviewer (prevents listening to its own trailing audio).
          // The generation tag lets the server reject stale reports.
          sendEvent({
            type: "playback_idle",
            generation: currentTurnGen != null ? currentTurnGen : 0,
            client_first_audio_ms: firstAudibleMs,
          });
        }
      };
    },
    (err) => {
      console.error("Audio decode failed", err);
      // A failed decode would leave playback never "finishing" server-side
      // and could wedge the session in speaking. Report idle instead.
      if (activeSources.length === 0) {
        sendEvent({
          type: "playback_idle",
          generation: currentTurnGen != null ? currentTurnGen : 0,
          client_first_audio_ms: firstAudibleMs,
        });
      }
    }
  );
}

function stopPlayback() {
  activeSources.forEach((source) => {
    try { source.stop(); } catch { /* already ended */ }
  });
  activeSources = [];
  nextPlayTime = 0;
}

// --------------------------------------------------------------------------
// UI helpers
// --------------------------------------------------------------------------
function setStatus(state) {
  els.statusChip.className = "chip state-" + state;
  els.statusChip.textContent = STATE_LABELS[state] || state;
}

function onInterviewStarted(msg) {
  startedAt = Date.now();
  answerEndClientTs = performance.now();
  clearInterval(timerInterval);
  timerInterval = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startedAt) / 1000);
    const m = String(Math.floor(elapsed / 60)).padStart(2, "0");
    const s = String(elapsed % 60).padStart(2, "0");
    els.timer.textContent = m + ":" + s;
  }, 1000);

  addToast(
    "info",
    "Interview started with " + msg.model + ". Answer whenever you're ready — just speak."
  );

  els.startButton.disabled = true;
  els.endButton.disabled = false;
  els.resetButton.disabled = false;
  els.nameInput.disabled = true;
  els.durationSelect.disabled = true;
  els.questionSelect.disabled = true;
}

function addBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = "bubble " + role;
  bubble.textContent = text;
  els.chat.appendChild(bubble);
  els.chat.scrollTop = els.chat.scrollHeight;
  return bubble;
}

function addNotice(text) {
  const toast = document.createElement("div");
  toast.className = "toast info";
  toast.textContent = text;
  els.chat.appendChild(toast);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function addToast(kind, text) {
  const toast = document.createElement("div");
  toast.className = "toast " + kind;
  toast.textContent = text;
  els.chat.appendChild(toast);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function appendAssistantText(text) {
  if (!assistantBubble) {
    assistantBubble = document.createElement("div");
    assistantBubble.className = "bubble assistant";
    els.chat.appendChild(assistantBubble);
  }
  assistantBubble.textContent += text;
  els.chat.scrollTop = els.chat.scrollHeight;
}

function updateStatePanel(msg) {
  els.statePanel.classList.remove("hidden");
  els.spPhase.textContent = msg.stage || "–";
  els.spHints.textContent =
    msg.hints + (msg.hint_level ? ` (level ${msg.hint_level})` : "");
  els.spAction.textContent =
    (msg.action || "–") + (msg.comment ? ` — ${msg.comment}` : "");
  if (msg.complexity && (msg.complexity.time || msg.complexity.status)) {
    els.spComplexity.textContent =
      `${msg.complexity.time || "?"} / ${msg.complexity.space || "?"} ` +
      `(${msg.complexity.status || "assessed"})`;
  } else {
    els.spComplexity.textContent = "–";
  }
  els.spIssues.textContent = (msg.issues || []).slice(-2).join("; ");
  if (msg.action === "finish_interview") {
    addNotice(
      "The interviewer thinks the interview can be wrapped up. Use ⏹ End interview for your evaluation."
    );
  }
}

// --------------------------------------------------------------------------
// Buttons
// --------------------------------------------------------------------------
els.startButton.addEventListener("click", async () => {
  const payload = {
    type: "start",
    question: els.questionSelect.value,
    name: els.nameInput.value,
    duration: els.durationSelect.value,
  };

  connect();
  const waitOpen = () =>
    ws.readyState === WebSocket.OPEN
      ? Promise.resolve()
      : new Promise((resolve, reject) => {
          ws.onopen = resolve;
          setTimeout(() => reject(new Error("timeout")), 5000);
        });

  try {
    await waitOpen();
    await startMic();
    sendEvent(payload);
  } catch (err) {
    addToast("error", "Could not connect to the interview server.");
    return;
  }
});

els.endButton.addEventListener("click", () => {
  sendEvent({ type: "end" });
  els.endButton.disabled = true;
});

els.interruptButton.addEventListener("click", () => {
  sendEvent({ type: "barge_in" });
  els.interruptButton.disabled = true;
});

els.resetButton.addEventListener("click", () => location.reload());

populateQuestions();
