"""Dialog-RSN-1 voice agent, following the "Python from scratch" guide.

This is the guide's full example with the fixes it needs to actually run.
Every deviation from the published code is marked FIX: and explained in README.md.
"""
import argparse
import asyncio
import base64
import json
import os

import sounddevice as sd
import websockets

# FIX: the guide hardcodes "wss://<your-realtime-host>/v1/realtime", which is not a
# resolvable host. Read it from the environment so the script runs as written.
URL = os.environ.get("DIALOGUE_URL", "wss://api.dev.poly.ai/v1/realtime")
API_KEY = os.environ["DIALOGUE_API_KEY"]
SAMPLE_RATE = 16000  # Dialog-RSN-1's native rate, so audio.input.format.rate is omitted

# FIX: the server's voice-activity detector works on 32ms frames (see the reference's
# turn-taking section: "five 32ms frames"). Left unset, sounddevice picks the device
# default, which on macOS is 15 samples (0.94ms) and yields ~1065 WebSocket messages
# per second for 16kHz audio: 3x more JSON than PCM.
BLOCKSIZE = 512  # 512 samples @ 16kHz = 32ms

TOOLS = {
    "get_time": lambda args: {"time": "11:04 AM", "timezone": "CDT"},
    "get_weather": lambda args: {
        "location": args["location"],
        "temperature_f": 68,
        "condition": "partly cloudy",
    },
}

SESSION_TOOLS = [
    {"type": "function", "name": "get_time", "description": "Get the current time",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "get_weather", "description": "Get the current weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                    "required": ["location"]}},
]


async def handle_tool_call(ws, name, arguments_json, call_id):
    arguments = json.loads(arguments_json) if arguments_json else {}
    # FIX: the guide's TOOLS[name] raises KeyError on a tool the model invents,
    # which kills the event loop. Answer the call instead so the turn can finish.
    tool = TOOLS.get(name)
    result = tool(arguments) if tool else {"error": f"unknown tool {name!r}"}
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": json.dumps(result)},
    }))
    await ws.send(json.dumps({"type": "response.create"}))


async def send_audio_task(ws, mic_queue):
    while True:
        frame = await mic_queue.get()
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(frame).decode(),
        }))


async def event_handler_task(ws, stop_after_turn=False):
    """FIX: the guide's version loops on `async for raw in ws` with no exit, so
    asyncio.wait(FIRST_COMPLETED) never actually fires and Ctrl-C is the only way
    out. stop_after_turn lets a one-shot text turn return instead of hanging."""
    reply = ""
    pending_tool = False
    async for raw in ws:
        event = json.loads(raw)
        event_type = event["type"]
        if event_type == "response.output_text.delta":
            reply += event["delta"]
        elif event_type == "response.output_text.done":
            print(f"agent: {reply}")
            reply = ""
        elif event_type == "response.function_call_arguments.done":
            print(f"  [tool] {event['name']}({event['arguments']})")
            await handle_tool_call(ws, event["name"], event["arguments"], event["call_id"])
        elif event_type == "input_audio_buffer.speech_started":
            print("(listening...)")
        # FIX: the guide never shows the transcript, so you get a reply with no record
        # of what the server heard you say. It arrives as part of the same generation.
        elif event_type == "conversation.item.input_audio_transcription.completed":
            print(f"you: {event.get('transcript', '')}")
        # FIX: the guide omits the error branch its own quickstart calls "not optional
        # padding". Every failure answers with an error event and LEAVES THE SOCKET
        # OPEN, so without this a failed turn just prints nothing and hangs forever.
        elif event_type == "error":
            raise RuntimeError(f"{event['error'].get('code')}: {event['error'].get('message')}")
        elif event_type == "response.done":
            status = event["response"]["status"]
            if status == "failed":
                raise RuntimeError(f"response failed: {event['response']}")
            if stop_after_turn and status == "completed" and not pending_tool:
                return
        # a tool call means another response follows, so the turn is not over yet
        if event_type == "response.function_call_arguments.done":
            pending_tool = True
        elif event_type == "response.output_text.done":
            pending_tool = False


def start_microphone(mic_queue, loop):
    def callback(indata, frames, time_info, status):
        pcm16 = (indata * 32767).astype("int16").tobytes()
        loop.call_soon_threadsafe(mic_queue.put_nowait, pcm16)

    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                          blocksize=BLOCKSIZE, callback=callback)


async def send_text_turn(ws, text):
    """Add-then-respond, from the quickstart. Handy for exercising tool calls
    without a microphone: the guide's own example has no text path at all."""
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": text}]},
    }))
    await ws.send(json.dumps({"type": "response.create"}))


async def main():
    parser = argparse.ArgumentParser(description="Dialog-RSN-1 voice agent")
    parser.add_argument("--say", help="send one text turn instead of opening the mic")
    args = parser.parse_args()

    headers = {"X-API-KEY": API_KEY}
    async with websockets.connect(URL, additional_headers=headers) as ws:
        await ws.recv()  # session.created

        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "instructions": "You are a helpful voice assistant. Keep responses brief.",
                "output_modalities": ["text"],
                "tools": SESSION_TOOLS,
                "audio": {"input": {"format": {"type": "audio/pcm"}}},
            },
        }))
        await ws.recv()  # session.updated

        if args.say:
            await send_text_turn(ws, args.say)
            await event_handler_task(ws, stop_after_turn=True)
            return

        mic_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        mic = start_microphone(mic_queue, loop)
        mic.start()
        print(f"connected to {URL}; speak into the mic, Ctrl-C to stop")

        tasks = [
            asyncio.create_task(send_audio_task(ws, mic_queue)),
            asyncio.create_task(event_handler_task(ws)),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in tasks:            # surface the exception that ended the race
                if task.done() and not task.cancelled():
                    task.result()
        finally:
            mic.stop()
            mic.close()  # FIX: the guide stops the stream but never releases the device
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
