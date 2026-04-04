import json
import os
import re
import signal
import sys
import threading
import time

import numpy as np
import pyaudio
import requests
import websocket
from openai import OpenAI

# ================= CONFIG =================
OPENAI_MODEL = "gpt-4.1-mini"

# Higgs vLLM server
HIGGS_API_BASE = "http://127.0.0.1:8002/v1"
HIGGS_MODEL = "higgs-audio-v2-generation-3B-base"
HIGGS_VOICE = "en_woman"  # Change if your server has other voice presets

# VTube Studio
VTS_WS = "ws://127.0.0.1:8001"
MOUTH_PARAMETER = "ParamMouthOpenY"

# Audio format from Higgs audio speech API example
AUDIO_RATE = 24000
AUDIO_CHANNELS = 1
AUDIO_FORMAT = pyaudio.paInt16
CHUNK_BYTES = 4096

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= GLOBALS =================
ws = None
ws_lock = threading.Lock()
state_lock = threading.Lock()

stop_speaking = False
current_thread = None


# ================= CLEAN EXIT =================
def shutdown(sig=None, frame=None):
    global ws, stop_speaking
    print("\n🛑 Shutting down...")

    with state_lock:
        stop_speaking = True

    try:
        with ws_lock:
            if ws is not None:
                ws.close()
                ws = None
    except Exception:
        pass

    sys.exit(0)


signal.signal(signal.SIGINT, shutdown)


# ================= LLM =================
def ask_llm(text: str) -> str:
    res = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a sarcastic anime VTuber. Reply in one short expressive sentence."
            },
            {"role": "user", "content": text}
        ]
    )
    reply = res.choices[0].message.content.strip()
    print("🤖", reply)
    return reply


# ================= TEXT =================
def split_sentences(text: str):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def is_valid_speech(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


# ================= VTS =================
def connect_vts():
    global ws

    try:
        print("🔌 Connecting VTS...")
        new_ws = websocket.create_connection(VTS_WS, timeout=5)

        token_req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(time.time()),
            "messageType": "AuthenticationTokenRequest",
            "data": {
                "pluginName": "AI VTuber Realtime",
                "pluginDeveloper": "OpenAI User"
            }
        }
        new_ws.send(json.dumps(token_req))
        token_res = json.loads(new_ws.recv())
        token = token_res["data"]["authenticationToken"]

        print("🔑 CLICK ALLOW IN VTS")

        auth_req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(time.time()),
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": "AI VTuber Realtime",
                "pluginDeveloper": "OpenAI User",
                "authenticationToken": token
            }
        }
        new_ws.send(json.dumps(auth_req))
        auth_res = json.loads(new_ws.recv())

        if auth_res.get("data", {}).get("authenticated"):
            with ws_lock:
                ws = new_ws
            print("✅ VTS CONNECTED")
            return ws

        print("❌ VTS auth failed")
        new_ws.close()
        return None

    except Exception as e:
        print("❌ VTS FAIL:", e)
        return None


def ws_send(payload):
    global ws

    for _ in range(3):
        try:
            if ws is None:
                if connect_vts() is None:
                    time.sleep(0.25)
                    continue

            with ws_lock:
                ws.send(json.dumps(payload))
                return True

        except Exception:
            print("⚠️ reconnecting VTS...")
            try:
                with ws_lock:
                    if ws is not None:
                        ws.close()
            except Exception:
                pass
            ws = None

    return False


def trigger_expression(hotkey_id: str):
    ws_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(time.time()),
        "messageType": "HotkeyTriggerRequest",
        "data": {"hotkeyID": hotkey_id}
    })


def send_mouth(value: float):
    value = max(0.0, min(float(value), 1.0))
    ws_send({
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(time.time()),
        "messageType": "InjectParameterDataRequest",
        "data": {
            "faceFound": True,
            "mode": "set",
            "parameterValues": [
                {"id": MOUTH_PARAMETER, "value": value}
            ]
        }
    })


def detect_emotion(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["haha", "lol", "hehe", "lmao"]):
        return "Smile"
    if any(x in t for x in ["what", "huh", "really", "whoa"]):
        return "Surprised"
    if any(x in t for x in ["ugh", "annoying", "seriously", "idiot"]):
        return "Angry"
    if any(x in t for x in ["sorry", "sad", "miss"]):
        return "Sad"
    return "Neutral"


# ================= HIGGS REALTIME =================
def pcm16le_to_float32(pcm_bytes: bytes) -> np.ndarray:
    if not pcm_bytes:
        return np.array([], dtype=np.float32)
    audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    return audio_i16.astype(np.float32) / 32768.0


def compute_mouth_from_pcm(pcm_bytes: bytes) -> float:
    samples = pcm16le_to_float32(pcm_bytes)
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples * samples)))
    return min(rms * 18.0, 1.0)


def higgs_stream_tts(text: str):
    """
    Streams raw PCM audio from Higgs vLLM audio speech API.
    """
    url = f"{HIGGS_API_BASE}/audio/speech"
    payload = {
        "model": HIGGS_MODEL,
        "voice": HIGGS_VOICE,
        "input": text,
        "response_format": "pcm"
    }

    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_content(chunk_size=CHUNK_BYTES):
                if chunk:
                    yield chunk
    except requests.RequestException as e:
        print("❌ Higgs realtime TTS failed:", e)
        return


def play_higgs_stream(text: str):
    global stop_speaking

    p = pyaudio.PyAudio()
    stream = None

    try:
        print("🔊 Realtime Higgs:", text)

        stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            output=True,
            frames_per_buffer=1024
        )

        prev_mouth = 0.0

        for pcm_chunk in higgs_stream_tts(text):
            with state_lock:
                if stop_speaking:
                    break

            stream.write(pcm_chunk)

            mouth = compute_mouth_from_pcm(pcm_chunk)
            mouth = prev_mouth * 0.65 + mouth * 0.35
            prev_mouth = mouth
            send_mouth(mouth)

        send_mouth(0.0)

    except Exception as e:
        print("❌ Playback failed:", e)
    finally:
        try:
            if stream is not None:
                stream.stop_stream()
                stream.close()
        except Exception:
            pass
        p.terminate()


# ================= SPEECH =================
def speak(text: str):
    print("🧠", text)

    # Keep whole reply together for lower overhead
    if not is_valid_speech(text):
        return

    play_higgs_stream(text)


def speak_async(text: str):
    global current_thread, stop_speaking

    with state_lock:
        stop_speaking = True

    if current_thread and current_thread.is_alive():
        current_thread.join(timeout=0.5)

    with state_lock:
        stop_speaking = False

    current_thread = threading.Thread(target=speak, args=(text,), daemon=True)
    current_thread.start()


# ================= MAIN =================
def main():
    print("🚀 VTuber starting...")
    connect_vts()

    while True:
        try:
            text = input("💬 ").strip()
            if not text:
                continue

            reply = ask_llm(text)

            if ws is not None:
                trigger_expression(detect_emotion(reply))

            speak_async(reply)

        except KeyboardInterrupt:
            shutdown()
        except Exception as e:
            print("❌", e)
            time.sleep(1)


if __name__ == "__main__":
    main()