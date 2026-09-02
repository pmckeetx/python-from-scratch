# Dialog-RSN-1 Python voice agent

Built by following [Build a voice agent from scratch with Python](https://dialog-rsn-1-eap-docs.pages.dev/dialog-rsn-1/guides/python-from-scratch/).

A terminal voice agent on the standard `websockets` package, no SDK: microphone in,
server-side turn detection, tool calling, reply streamed back as text.

## Running it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add your DIALOGUE_API_KEY
set -a && . ./.env && set +a

python agent.py                                   # open the mic and talk
python agent.py --say "What time is it right now?" # one text turn, no mic needed
```

Needs Python 3.10+ (`websockets>=14` renamed `extra_headers` to `additional_headers`).

## Layout

| File | What it is |
|---|---|
| `agent.py` | The guide's agent with the fixes below. Every deviation is marked `FIX:`. |
| `agent_verbatim.py` | The guide's full example byte-for-byte, to reproduce the failures. |

## Findings

Verified against `wss://api.dev.poly.ai/v1/realtime` on 2026-09-02.

### 1. `pip install websockets sounddevice` leaves out numpy

Blocker. The guide's install line is missing `numpy`, and the failure is not the subtle kind
you'd hope for. `sd.InputStream` raises at construction, before any audio flows:

```
FAILED at mic construction/start: ModuleNotFoundError: No module named 'numpy'
```

`sounddevice` declares only `CFFI`, so nothing pulls numpy in transitively, and the guide's
`(indata * 32767).astype("int16")` needs it anyway. The install line should be
`pip install 'websockets>=14' sounddevice numpy`.

### 2. `wss://<your-realtime-host>/v1/realtime` is a placeholder

Nothing in the guide or the quickstart says what the host is, so the copied script fails at
`connect()`. The real one is `wss://api.dev.poly.ai/v1/realtime`; `agent.py` reads
`DIALOGUE_URL` and defaults to it.

### 3. "Three concurrent tasks" — there are two

Guide line 63: "A real agent needs to send audio, watch for a turn to start, and process
incoming events, all at once. Three concurrent tasks, raced rather than run in sequence."
The code below it creates two: `send_audio_task` and `event_handler_task`. Watching for a
turn to start is not a task, it's a branch inside the event handler. Either the prose should
say two, or a third task is missing from the example.

### 4. No `error` branch, which the guide's own quickstart calls mandatory

`event_handler_task` handles four event types and `error` isn't one of them. The quickstart is
unambiguous about why that matters: every failure "answers with an `error` event and **leaves
the connection open**", so "skip the branch and a failed generation prints an empty string and
exits zero." In the audio agent it's worse than that, because nothing terminates the loop, so a
failed turn leaves the process sitting there silently forever. Same for
`response.done(status: "failed")`.

### 5. No `blocksize`, so macOS sends ~1065 WebSocket messages a second

Left unset, `sounddevice` takes the device default, which here is **15 samples — 0.94ms** per
callback. Measured:

```
callbacks in 1s: 1065
samples per callback: [15]
=> one ws message = 90 bytes, ~1065 messages/sec
=> 94 KiB/s of JSON for 31 KiB/s of audio
```

Three times more protocol than payload. The reference specifies the detector's framing
("five 32ms frames"), so `blocksize=512` matches it: 158 frames for a 2s utterance instead
of 5370. Both work — the server accepted the 15-sample frames and transcribed correctly —
but the guide is teaching the pathological default.

### 6. The task race never fires, so `asyncio.wait` is decorative

`event_handler_task` loops on `async for raw in ws` with no exit path, and `send_audio_task`
loops forever. Neither task can complete, so `asyncio.wait(..., return_when=FIRST_COMPLETED)`
only returns when something raises. The prose (line 102) sells the pattern as what "keeps a
clean Ctrl-C from turning into a pile of dangling tasks", but Ctrl-C is the *only* way out.

### 7. `asyncio.wait` doesn't re-raise, so a crashed task disappears

Following on from #6: when a task does raise, `asyncio.wait` returns normally rather than
propagating, unlike `asyncio.gather`. The guide never calls `task.result()`, so the `finally`
block cancels everything and `main()` returns cleanly on what was actually a crash. `agent.py`
calls `.result()` on the completed task to surface it.

### 8. `TOOLS[name]` raises `KeyError` on a tool the model invents

One hallucinated tool name takes down the event loop, and per #7 it does so silently. Answering
the call with an error payload keeps the conversation alive.

### 9. `mic.stop()` never releases the device

The `finally` block stops the stream but never calls `mic.close()`, so the input device stays
claimed until the process exits.

### 10. The intro promises a text path the example doesn't have

"It starts on text turns, the same add-then-respond shape from the quickstart, then adds tool
calling and, finally, real audio input." The full example is audio-only — there is no
`conversation.item.create` with `input_text` anywhere in it. So the one part a reader can test
without a working microphone, tool calling, is the part they can't reach. `agent.py --say`
adds it back in six lines.

## Confirmed accurate

- **Omitting `turn_detection` is fine.** The guide's `session.update` leaves it out; the
  server's default is `{"type": "server_vad", "create_response": true}`, so turns answer
  themselves. Verified on the `session.created` echo.
- **Tool calling works exactly as written.** `handle_tool_call` and the
  `response.function_call_arguments.done` shape are correct against the live API, both tools:

  ```
  [tool] get_time({})                      -> agent: It's 11:04 AM CDT.
  [tool] get_weather({"location": "Tokyo"}) -> agent: It's 68 degrees and partly cloudy in Tokyo.
  ```

- **16000 really is native**, and omitting `audio.input.format.rate` is right.
- **Full acoustic loop works** once numpy and the host are sorted: mic -> `speech_started` ->
  transcript -> tool call -> reply.

## One inconsistency in the reference, not this guide

The reference's session table describes `poly_input_rate` as "A real client sample rate outside
the standard **24 kHz**", two rows after `audio.input.format` says to omit `rate` "to get
Dialog-RSN-1's native **16 kHz**". This guide says 16kHz and is right; the 24kHz mention looks
like a leftover from the OpenAI-shaped original.
