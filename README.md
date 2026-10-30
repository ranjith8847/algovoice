# Real-time DSA Voice Interviewer (Phase 1)

Hands-free voice interviews: speak naturally, no buttons. The interviewer
listens continuously (VAD), transcribes locally (faster-whisper), reasons
locally (LM Studio), and speaks sentence-by-sentence (Piper) — all on your
Mac. Start talking over the interviewer's voice and it stops and listens
(bar-in).

## Run it

```bash
# 1. LM Studio: start the server (default port 1234) and load a model.

# 2. From the project root:
source ai_dsa_env/bin/activate
python -m interviewer.app

# 3. Open http://127.0.0.1:8100 in Chrome or Safari (mic permission required).
```

First run downloads the Piper voice (~60 MB, cached in `models/piper/`) and
possibly the Whisper model (cached in `~/.cache/huggingface`).

Click **Start interview** and just talk. Pause to think as long as you like
(silence tolerance ~1.1 s by default). Click **End interview and get
feedback** for the spoken evaluation.

## Architecture

```
Browser (index.html / app.js)
  getUserMedia(16 kHz, echoCancellation) --> PCM16 chunks --> WebSocket
  <-- state / transcripts / streamed text / sentence WAVs <-- WebSocket
        |
        v
FastAPI  (interviewer/app.py)
  InterviewSession  (interviewer/controller.py)  <-- state machine:
  | IDLE -> LISTENING --(speech end)--> PROCESSING -> THINKING
  | -> SPEAKING --> LISTENING ... (barge-in cancels generation)
  |
  +-- webrtcvad      : 30 ms frames, speech onset/offset detection
  +-- faster-whisper : transcribes the completed utterance (asr.py)
  +-- LM Studio      : streams tokens, split into sentences (llm_client.py)
  +-- Piper          : synthesizes each sentence to WAV (tts.py)
```

Timeout you'll see in the UI after each answer: *first token X ms ·
first audio Y ms after your answer* — the puck from the latency plan
(t0 speech-end → t2 first LLM token → t4 first audible audio).

## Phase 4: the interview intelligence layer

The interviewer is now interview-aware, not just transcript-reactive:

- **Structured interview state** (`interviewer/interview_state.py`): stage
  (introduction → understanding → algorithm → complexity → edge cases →
  code review → evaluation), approach notes, complexity assessment, open
  issues, and progressive-hint counters. Phases are advisory — they advance
  on the decision layer's suggestion, never regress silently.
- **Decision layer** (`interviewer/decision.py`): after each spoken turn, a
  background non-streaming LM Studio call returns a Pydantic-validated JSON
  assessment (understanding, issues, complexity status, interviewer_action,
  next question, hint level). It never adds latency to the voice loop — it
  runs *after* the reply finishes speaking and shapes the *next* turn.
  Invalid output is logged and skipped; assessments are evidence, not
  ground truth.
- **System-prompt directive**: the tracker is baked into the (always-first)
  system message each turn — stage, hint policy, open issues, code-pasted
  flag — so streaming replies adapt without an extra blocking call.
- **Progressive hints**: `give_hint` increments the hint counter/level; the
  directive tells the interviewer to go one level deeper instead of solving.
- **Final evaluation** is grounded in the tracker (independence level vs
  hints used, phases covered, complexity assessments) in addition to the
  transcript.
- **UI**: a small tracker bar (stage · hints · last action · complexity)
  updates after each answer; a `finish_interview` suggestion surfaces a
  notice so you know to click End interview.

Known scope (per the plan): code execution remains out (conceptual review
only, explicitly stated by the interviewer); `finish_interview` is a
suggestion, not an auto-stop.

## Tuning knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `END_OF_UTTERANCE_SILENCE_MS` | 1100 | silence that ends your utterance (raise if it cuts you off while thinking) |
| `MIN_UTTERANCE_MS` | 400 | ignore noises/shorter than this |
| `MAX_UTTERANCE_MS` | 60000 | force-submits long ongoing speech |
| `VAD_AGGRESSIVENESS` | 3 | 0–3, higher = fewer false speech detections |
| `AUTO_BARGE_IN` | 0 | set to 1 to interrupt the interviewer just by speaking (best with headphones + Chrome). Default off; use the ⏸ Interrupt button instead |
| `BARGE_IN_CONFIRM_MS` | 350 | sustained speech required before a VAD barge-in fires |
| `WHISPER_MODEL_SIZE` | small | base / small / medium / large-v3 |
| `TRANSCRIBE_LANGUAGE` | en | or two-letter code; `auto` risks misdetection |
| `LLM_MAX_TOKENS` | 1500 | raise if a thinking model returns empty replies |
| `DECISION_MAX_TOKENS` | 900 | token budget for the background assessment call |
| `PIPER_VOICE_NAME` | en_US-amy-medium | any voice from rhasspy/piper-voices |
| `PIPER_LENGTH_SCALE` | 1.0 | >1.0 speaks slower, <1.0 faster |
| `LM_STUDIO_MODEL` | (auto) | pin the model ID; else first model listed by LM Studio |

Example: `END_OF_UTTERANCE_SILENCE_MS=1400 python -m interviewer.app`

## What Phase 1 includes (and what it does not yet)

Included:
- Continuous listening with VAD end-of-utterance detection
- Hands-free loop: speak → transcribe → reply streams in → spoken sentence-by-sentence
- Interruption: the ⏸ button stops playback and generation instantly;
  optional voice-triggered barge-in (`AUTO_BARGE_IN=1`). Cancelled turns use
  a generation ID so stale output never plays after a new question, and
  partially spoken replies are preserved in history marked as
  "[interrupted by the candidate]"
- Echo protection: the server keeps the SPEAKING state until the browser
  reports playback finished, and pauses mic capture while the interviewer
  audio is playing (unless auto barge-in is on)
- Latency measurements per answer
- Timer, question bank, final evaluation on end

Not yet (phases 2+ per the architecture plan):
- Semantic endpointing (understanding that a unfinished sentence isn't a question)
- Streaming per-token TTS and overlapping synthesis while the LLM still runs
  (currently: sentence-level, which is fast enough in practice: ~0.2 s/sentence)
- Interview persistence (SQLite), auto-end on duration expiry
- A hardened React frontend; the current page is intentionally small

## Confidence, headphones, and echo

The browser mic uses OS-level echo cancellation, which handles most cases on
Apple Silicon laptop speakers. If the interviewer ever reacts to its own
voice, switch to headphones or raise `VAD_AGGRESSIVENESS` to keep 3.

## Files

```
interviewer/
  config.py       every tunable in one place
  questions.py    persona, question bank, greeting/feedback prompts
  llm_client.py   LM Studio streaming + sentence splitting + Qwen-safe normalization
  asr.py          faster-whisper over completed utterances
  tts.py          Piper voice loading + sentence WAV synthesis
  controller.py   InterviewSession state machine, VAD loop, pipeline relay
  app.py          FastAPI wiring, WebSocket endpoint, static hosting
  static/         index.html, app.js, style.css
scripts/smoke_test.py   end-to-end WebSocket test (greeting pipeline)
```
