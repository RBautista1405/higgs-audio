import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import subprocess
import json
import websocket
import time
import torch
import os
import threading
import re
import signal
import sys

from faster_whisper import WhisperModel
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= CONFIG =================
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.5
VTS_WS = "ws://localhost:8001"

# ================= GLOBAL STATE =================
state_lock = threading.Lock()
ws_lock = threading.Lock()

is_speaking = False
stop_speaking = False
current_speech_thread = None
ws = None

# ================= CUDA =================
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA not available")

print("🚀 Loading Whisper...")
whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
print("✅ Whisper ready")

# ================= CLEAN EXIT =================
def shutdown(sig=None, frame=None):
    print("\n🛑 Shutting down...")
    try:
        sd.stop()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

# ================= AUDIO =================
def record_audio(duration=1.5):
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()

    # 🔥 silence detection (prevents random triggers)
    if np.abs(audio).mean() < 100:
        return None

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp.name, SAMPLE_RATE, audio)
    return temp.name

# ================= STT =================
def transcribe(path):
    if not path:
        return ""

    segments, _ = whisper_model.transcribe(path)
    text = " ".join([s.text for s in segments]).strip()
    os.remove(path)

    if len(text) < 3:
        return ""

    print("📝 You:", text)
    return text

# ================= LLM =================
def ask_llm(text):
    res = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a cute sarcastic anime VTuber. Reply in one short expressive sentence."},
            {"role": "user", "content": text}
        ]
    )
    reply = res.choices[0].message.content.strip()
    print("🤖 AI:", reply)
    return reply

# ================= HIGGS =================
def higgs_tts(text):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    try:
        subprocess.run([
            "python", "higgs_audio/infer.py",
            "--text", text,
            "--output", out
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except:
        print("⚠️ Higgs failed")
        return None

    return out

def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# ================= VTS =================
def ws_send(payload):
    global ws

    try:
        with ws_lock:
            ws.send(json.dumps(payload))
            ws.settimeout(2)  # 🔥 prevent freeze
            return json.loads(ws.recv())
    except:
        print("⚠️ WS reconnecting...")
        ws = connect_vts()
        return None

def connect_vts():
    global ws

    while True:
        try:
            print("🔌 Connecting to VTube Studio...")
            ws = websocket.create_connection(VTS_WS, timeout=5)

            # 🔥 Always triggers permission popup
            token_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(time.time()),
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": "AI VTuber",
                    "pluginDeveloper": "You"
                }
            }

            ws.send(json.dumps(token_req))
            res = json.loads(ws.recv())

            if "data" not in res:
                print("⚠️ Waiting for VTS permission popup...")
                time.sleep(2)
                continue

            token = res["data"]["authenticationToken"]
            print("🔑 Token received → CLICK ALLOW IN VTS")

            auth_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": str(time.time()),
                "messageType": "AuthenticationRequest",
                "data": {
                    "pluginName": "AI VTuber",
                    "pluginDeveloper": "You",
                    "authenticationToken": token
                }
            }

            ws.send(json.dumps(auth_req))
            auth_res = json.loads(ws.recv())

            if auth_res.get("data", {}).get("authenticated"):
                print("✅ Connected to VTube Studio")
                return ws

            print("⚠️ Not authenticated yet...")
            time.sleep(2)

        except Exception as e:
            print("❌ VTS connection failed:", e)
            time.sleep(2)

# ================= EXPRESSIONS =================
def detect_emotion(text):
    t = text.lower()

    if any(x in t for x in ["haha", "lol", "hehe", "yay"]):
        return "Smile"
    if any(x in t for x in ["what", "huh", "really"]):
        return "Surprised"
    if any(x in t for x in ["ugh", "annoying", "seriously"]):
        return "Angry"
    if any(x in t for x in ["sorry", "sad", "miss"]):
        return "Sad"

    return "Neutral"

def trigger_expression(hotkey):
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(time.time()),
        "messageType": "HotkeyTriggerRequest",
        "data": {"hotkeyID": hotkey}
    }
    ws_send(msg)

# ================= MOUTH =================
def send_mouth(chunk):
    vol = np.abs(chunk).mean()
    mouth = min(vol / 2000, 1.0)

    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": str(time.time()),
        "messageType": "InjectParameterDataRequest",
        "data": {
            "parameterValues": [
                {"id": "MouthOpen", "value": float(mouth)}
            ]
        }
    }
    ws_send(msg)

# ================= PLAYBACK =================
def play_audio(path):
    global is_speaking, stop_speaking

    if not path:
        return

    rate, data = wav.read(path)

    if len(data.shape) > 1:
        data = data[:, 0]

    with state_lock:
        is_speaking = True
        stop_speaking = False

    sd.play(data, rate)

    chunk = 2048
    for i in range(0, len(data), chunk):

        with state_lock:
            if stop_speaking:
                sd.stop()  # 🔥 instant cut
                break

        send_mouth(data[i:i+chunk])
        time.sleep(chunk / rate)

    sd.stop()

    with state_lock:
        is_speaking = False

    os.remove(path)

# ================= SPEECH =================
def speak_stream(text):
    sentences = split_sentences(text)

    for s in sentences:
        with state_lock:
            if stop_speaking:
                break

        if not s.strip():
            continue

        audio = higgs_tts(s)
        play_audio(audio)

def speak_async(text):
    global current_speech_thread

    # 🔥 prevent stacking threads
    if current_speech_thread and current_speech_thread.is_alive():
        return

    current_speech_thread = threading.Thread(
        target=speak_stream,
        args=(text,),
        daemon=True
    )
    current_speech_thread.start()

# ================= MAIN =================
def main():
    global stop_speaking

    print("🚀 Starting VTuber...")
    connect_vts()

    while True:
        try:
            audio = record_audio(CHUNK_DURATION)
            text = transcribe(audio)

            if not text:
                continue

            # 🔥 instant interrupt
            with state_lock:
                stop_speaking = True

            reply = ask_llm(text)

            emotion = detect_emotion(reply)
            trigger_expression(emotion)

            speak_async(reply)

        except Exception as e:
            print("❌ ERROR:", e)
            time.sleep(1)

# ================= RUN =================
if __name__ == "__main__":
    main()