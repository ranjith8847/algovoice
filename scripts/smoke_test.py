"""End-to-end smoke test for the interviewer WebSocket flow.

Boots against a running server on 127.0.0.1:8100 and exercises the
start -> greeting pipeline. Run:
    ai_dsa_env/bin/python scripts/smoke_test.py
"""

import asyncio
import json
import time

import websockets


async def main() -> None:
    uri = "ws://127.0.0.1:8100/ws"
    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        first = await ws.recv()
        print("on connect:", first)

        await ws.send(
            json.dumps(
                {
                    "type": "start",
                    "question": "Two Sum (Easy)",
                    "name": "Test",
                    "duration": 30,
                }
            )
        )
        t0 = time.perf_counter()
        text_events = []
        audio_bytes = 0
        sent_playback_idle = False

        deadline = t0 + 40
        while time.perf_counter() < deadline:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=35)
            except asyncio.TimeoutError:
                print("timeout waiting for messages")
                break
            if isinstance(msg, bytes):
                audio_bytes += len(msg)
                continue
            data = json.loads(msg)
            kind = data.get("type")
            if kind == "assistant_delta":
                text_events.append(data["text"])
            elif kind != "assistant_delta":
                elapsed = time.perf_counter() - t0
                print(f"{elapsed:6.1f}s", kind, json.dumps(data)[:200])
            if (
                data.get("type") == "state"
                and data.get("state") == "speaking"
                and not sent_playback_idle
            ):
                # Simulate the browser finishing playback of all sentences.
                sent_playback_idle = True
                await asyncio.sleep(2)
                await ws.send(json.dumps({"type": "playback_idle"}))
            if (
                data.get("type") == "state"
                and data.get("state") == "listening"
            ):
                break

        text = "".join(text_events)
        print("ASSISTANT TEXT:", (text[:300] + "…") if len(text) > 300 else text)
        print("AUDIO BYTES:", audio_bytes)


if __name__ == "__main__":
    asyncio.run(main())
