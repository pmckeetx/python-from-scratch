import asyncio
import base64
import json
import os

import sounddevice as sd
import websockets

URL = "wss://<your-realtime-host>/v1/realtime"
API_KEY = os.environ["DIALOGUE_API_KEY"]
SAMPLE_RATE = 16000

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
    result = TOOLS[name](arguments)
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


async def event_handler_task(ws):
    reply = ""
    async for raw in ws:
        event = json.loads(raw)
        event_type = event["type"]
        if event_type == "response.output_text.delta":
            reply += event["delta"]
        elif event_type == "response.output_text.done":
            print(reply)
            reply = ""
        elif event_type == "response.function_call_arguments.done":
            await handle_tool_call(ws, event["name"], event["arguments"], event["call_id"])
        elif event_type == "input_audio_buffer.speech_started":
            print("(listening...)")


def start_microphone(mic_queue, loop):
    def callback(indata, frames, time_info, status):
        pcm16 = (indata * 32767).astype("int16").tobytes()
        loop.call_soon_threadsafe(mic_queue.put_nowait, pcm16)

    return sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", callback=callback)


async def main():
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

        mic_queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        mic = start_microphone(mic_queue, loop)
        mic.start()

        tasks = [
            asyncio.create_task(send_audio_task(ws, mic_queue)),
            asyncio.create_task(event_handler_task(ws)),
        ]
        try:
            await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            mic.stop()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(main())
