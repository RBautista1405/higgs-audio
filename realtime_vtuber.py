import numpy as np
import scipy.io.wavfile as wav
import tempfile
import subprocess
import json
import websocket
import time
import os
import threading
import re
import signal
import sys

from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= CONFIG =================
VTS_WS = "ws://127.0.0.1:8001"

# 👉 CHANGE THIS if needed
MODEL_PATH = "model"  # e.g. "checkpoints/higgs.pt"

# ================= GLOBAL STATE =================
state_lock = threading.Lock()
ws_lock = threading.Lock()

is_speaking = False
stop_speaking = False
current_speech_thread = None
ws = None

# ================= CLEAN EXIT =================
def shutdown(sig=None, frame=None):
    print("\n🛑 Shutting down...")
    try:
        import sounddevice as sd
        sd.stop()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)

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

# ================= HIGGS (FIXED) =================
def higgs_tts(text):
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

    print("🔊 Generating TTS (Higgs real)...")

    try:
        subprocess.run([
            "python",
            "examples/generation.py",
            "--transcript", text,
            "--out_path", out
        ], check=True)
    except Exception as e:
        print("❌ Higgs failed:", e)
        return None

    print(f"📁 TTS file: {out}")
    return out

def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)

# ================= VTS =================
def ws_send(payload):
    global ws

    try:
        with ws_lock:
            ws.send(json.dumps(payload))
            ws.settimeout(2)
            return json.loads(ws.recv())
    except:
        print("⚠️ WS reconnecting...")
        return None

def connect_vts():
    global ws

    try:
        print("🔌 Connecting to VTube Studio...")
        ws = websocket.create_connection(VTS_WS, timeout=5)

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
            print("⚠️ No token received")
            return None

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

        print("⚠️ Authentication failed")
        return None

    except Exception as e:
        print("❌ VTS connection failed:", e)
        return None

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
    # Convert to float32 in range -1.0 to 1.0
    if chunk.dtype.kind == 'i':
        chunk = chunk.astype(np.float32) / np.iinfo(chunk.dtype).max
    elif chunk.dtype.kind == 'f':
        chunk = chunk.astype(np.float32)

    vol = np.abs(chunk).mean()
    # Scale volume to mouth open (tweak 0.2 multiplier for sensitivity)
    mouth = min(vol * 3.0, 1.0)  

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
        print("❌ No audio file")
        return

    import sounddevice as sd

    rate, data = wav.read(path)

    if len(data.shape) > 1:
        data = data[:, 0]

    print("🔊 Playing audio...")

    with state_lock:
        is_speaking = True
        stop_speaking = False

    sd.play(data, rate)

    chunk = 2048
    for i in range(0, len(data), chunk):

        with state_lock:
            if stop_speaking:
                sd.stop()
                break

        send_mouth(data[i:i+chunk])
        time.sleep(chunk / rate)

    sd.stop()

    with state_lock:
        is_speaking = False

    os.remove(path)

# ================= SPEECH =================
def speak_stream(text):
    print("🧠 Speaking:", text)

    sentences = split_sentences(text)

    for s in sentences:
        if not s.strip():
            continue

        print("➡️ TTS sentence:", s)

        audio = higgs_tts(s)

        if not audio:
            print("❌ No audio generated")
            continue

        play_audio(audio)

def speak_async(text):
    global current_speech_thread, stop_speaking

    print("🚀 Starting speech thread")

    with state_lock:
        stop_speaking = True

    time.sleep(0.1)

    current_speech_thread = threading.Thread(
        target=speak_stream,
        args=(text,),
        daemon=True
    )
    current_speech_thread.start()

# ================= MAIN =================
def main():
    print("🚀 Starting VTuber (TEXT MODE)...")

    ws_conn = connect_vts()

    if ws_conn:
        print("🟢 Ready for input!")
    else:
        print("⚠️ Running WITHOUT VTube Studio")

    while True:
        try:
            text = input("💬 You: ").strip()

            if not text:
                continue

            reply = ask_llm(text)

            if ws_conn:
                emotion = detect_emotion(reply)
                trigger_expression(emotion)

            speak_async(reply)

        except Exception as e:
            print("❌ ERROR:", e)
            time.sleep(1)

# ================= RUN =================
if __name__ == "__main__":
    main()