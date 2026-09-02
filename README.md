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
| `agent.py` | The guide's current agent, plus a couple of local conveniences (a real default host, transcript/tool-call printing, Ctrl-C handling). |
| `agent_verbatim.py` | The guide's full example byte-for-byte as it read *before* the 2026-09-02 fix, kept to reproduce the findings below. |

## Findings

Verified against `wss://api.dev.poly.ai/v1/realtime` on 2026-09-02. Findings 1 and 3–10 were
fixed upstream in the guide the same day ([`dialog-rsn-1__eap_docs`](https://github.com/PolyAI-LDN/dialog-rsn-1__eap_docs),
`dialog-rsn-1/guides/python-from-scratch.mdx`); `agent_verbatim.py` reproduces them as they
originally read. Finding 2 is still open.

### 1. ~~`pip install websockets sounddevice` leaves out numpy~~ — fixed in the guide

Was a blocker. The guide's install line was missing `numpy`, and the failure was not the subtle
kind you'd hope for. `sd.InputStream` raised at construction, before any audio flowed:

```
FAILED at mic construction/start: ModuleNotFoundError: No module named 'numpy'
```

`sounddevice` declares only `CFFI`, so nothing pulls numpy in transitively, and the guide's
`(indata * 32767).astype("int16")` needs it anyway. The guide's install line now reads
`pip install websockets sounddevice numpy`.

### 2. `wss://<your-realtime-host>/v1/realtime` is a placeholder — still open

Nothing in the guide or the quickstart says what the host is, so the copied script fails at
`connect()`. This is the same placeholder used across every guide, the quickstart, and the
reference, so it's out of scope to fix in this one guide alone. `agent.py` reads `DIALOGUE_URL`
and defaults to the prod host, `wss://api.us.poly.ai/v1/realtime`; set `DIALOGUE_URL` to point
at dev (`wss://api.dev.poly.ai/v1/realtime`, what the findings below were verified against)
instead.

### 3. ~~"Three concurrent tasks" — there were two~~ — fixed in the guide

Guide line 63 used to read: "A real agent needs to send audio, watch for a turn to start, and
process incoming events, all at once. Three concurrent tasks, raced rather than run in
sequence." The code below it created two: `send_audio_task` and `event_handler_task`. Watching
for a turn to start was never a task, it's a branch inside the event handler. The prose now says
two.

### 4. ~~No `error` branch, which the guide's own quickstart calls mandatory~~ — fixed in the guide

`event_handler_task` handled four event types and `error` wasn't one of them. The quickstart is
unambiguous about why that matters: every failure "answers with an `error` event and **leaves
the connection open**", so "skip the branch and a failed generation prints an empty string and
exits zero." In the audio agent it was worse than that, because nothing terminated the loop, so
a failed turn left the process sitting there silently forever. Same for
`response.done(status: "failed")`. The guide's `event_handler_task` now raises on both.

### 5. ~~No `blocksize`, so macOS sends ~1065 WebSocket messages a second~~ — fixed in the guide

Left unset, `sounddevice` took the device default, which here was **15 samples — 0.94ms** per
callback. Measured:

```
callbacks in 1s: 1065
samples per callback: [15]
=> one ws message = 90 bytes, ~1065 messages/sec
=> 94 KiB/s of JSON for 31 KiB/s of audio
```

Three times more protocol than payload. The reference specifies the detector's framing
("five 32ms frames"), so `blocksize=512` matches it: 158 frames for a 2s utterance instead
of 5370. Both worked — the server accepted the 15-sample frames and transcribed correctly —
but the guide was teaching the pathological default. It now sets `BLOCKSIZE = 512`.

### 6. ~~The task race never fired, so `asyncio.wait` was decorative~~ — fixed in the guide

`event_handler_task` looped on `async for raw in ws` with no exit path, and `send_audio_task`
looped forever. Neither task could complete, so `asyncio.wait(..., return_when=FIRST_COMPLETED)`
only returned on Ctrl-C. The prose sold the pattern as what "keeps a clean Ctrl-C from turning
into a pile of dangling tasks", but Ctrl-C was the *only* way out. Now that `error` and
`response.done(status: "failed")` raise (#4), a failed turn ends the race too, which is what
the prose describes.

### 7. ~~`asyncio.wait` doesn't re-raise, so a crashed task disappeared~~ — fixed in the guide

Following on from #6: when a task raises, `asyncio.wait` returns normally rather than
propagating, unlike `asyncio.gather`. The guide never called `task.result()`, so the `finally`
block cancelled everything and `main()` returned cleanly on what was actually a crash. The guide
now calls `.result()` on each completed task to surface it.

### 8. ~~`TOOLS[name]` raised `KeyError` on a tool the model invents~~ — fixed in the guide

One hallucinated tool name took down the event loop, and per #7 it did so silently. The guide's
`handle_tool_call` now looks the tool up with `.get` and answers with an error payload instead,
keeping the conversation alive.

### 9. ~~`mic.stop()` never released the device~~ — fixed in the guide

The `finally` block stopped the stream but never called `mic.close()`, so the input device
stayed claimed until the process exited. The guide's `finally` block now calls both.

### 10. ~~The intro promised a text path the example didn't have~~ — fixed in the guide

"It starts on text turns, the same add-then-respond shape from the quickstart, then adds tool
calling and, finally, real audio input." The full example was audio-only — there was no
`conversation.item.create` with `input_text` anywhere in it. So the one part a reader could test
without a working microphone, tool calling, was the part they couldn't reach. The guide's full
example now takes a `--say` flag that sends one text turn instead of opening the mic.

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
