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

from faster_whisper import WhisperModel
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= CONFIG =================

SAMPLE_RATE = 16000
DURATION = 5
VTS_WS = "ws://localhost:8001"

# ================= CUDA CHECK =================
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA is NOT available. Fix your setup.")

print("🚀 Initializing Whisper on GPU...")

whisper_model = WhisperModel(
    "base",  # you can upgrade to "medium"
    device="cuda",
    compute_type="float16"
)

print("✅ Whisper GPU ready")

# ================= AUDIO =================
def record_audio(duration=5):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_file.name, SAMPLE_RATE, audio)

    return temp_file.name

# ================= STT =================
def transcribe(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = " ".join([seg.text for seg in segments])
    print("📝 You:", text)
    return text

# ================= LLM =================
def ask_llm(user_text):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a cute sarcastic anime VTuber. Keep replies short and expressive."},
            {"role": "user", "content": user_text}
        ]
    )

    reply = response.choices[0].message.content
    print("🤖 AI:", reply)
    return reply

# ================= HIGGS TTS =================
def higgs_tts(text):
    output_path = "output.wav"

    subprocess.run([
        "python", "higgs_audio/infer.py",
        "--text", text,
        "--output", output_path
    ], check=True)

    return output_path

# ================= VTS CONNECTION =================
def connect_vts():
    ws = websocket.create_connection(VTS_WS)

    print("🔌 Connecting to VTube Studio API...")

    # Request token
    token_request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "token_req",
        "messageType": "AuthenticationTokenRequest",
        "data": {
            "pluginName": "AI VTuber",
            "pluginDeveloper": "You"
        }
    }

    ws.send(json.dumps(token_request))
    response = json.loads(ws.recv())

    if "data" not in response:
        raise RuntimeError("❌ Failed to get token")

    token = response["data"]["authenticationToken"]
    print("🔑 Token received")

    # Authenticate
    auth_request = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "auth_req",
        "messageType": "AuthenticationRequest",
        "data": {
            "pluginName": "AI VTuber",
            "pluginDeveloper": "You",
            "authenticationToken": token
        }
    }

    ws.send(json.dumps(auth_request))
    auth_response = json.loads(ws.recv())

    if not auth_response.get("data", {}).get("authenticated", False):
        raise RuntimeError("❌ Authentication failed")

    print("✅ Connected & Authenticated with VTube Studio")

    return ws

# ================= EXPRESSIONS =================
def trigger_expression(ws, hotkey_id):
    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "exp_req",
        "messageType": "HotkeyTriggerRequest",
        "data": {
            "hotkeyID": hotkey_id
        }
    }

    ws.send(json.dumps(msg))

    try:
        response = json.loads(ws.recv())
        if response.get("messageType") == "HotkeyTriggerResponse":
            print(f"🎭 Expression: {hotkey_id}")
        else:
            print("⚠️ Unexpected VTS response:", response)
    except:
        print("⚠️ No response from VTS")

def detect_emotion(text):
    t = text.lower()

    if any(x in t for x in ["love", "yay", "happy", "hehe"]):
        return "Smile"
    elif any(x in t for x in ["angry", "hate", "annoying"]):
        return "Angry"
    elif any(x in t for x in ["sad", "sorry"]):
        return "Sad"

    return "Neutral"

# ================= MOUTH MOVEMENT =================
def send_mouth_movement(ws, audio_chunk):
    volume = np.abs(audio_chunk).mean()
    mouth_value = min(volume / 3000, 1.0)

    msg = {
        "apiName": "VTubeStudioPublicAPI",
        "apiVersion": "1.0",
        "requestID": "mouth_req",
        "messageType": "InjectParameterDataRequest",
        "data": {
            "parameterValues": [
                {"id": "MouthOpen", "value": float(mouth_value)}
            ]
        }
    }

    ws.send(json.dumps(msg))

# ================= AUDIO PLAYBACK =================
def play_audio_with_vts(ws, file_path):
    rate, data = wav.read(file_path)

    if len(data.shape) > 1:
        data = data[:, 0]

    sd.play(data, rate)

    chunk_size = 2048
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        send_mouth_movement(ws, chunk)
        time.sleep(chunk_size / rate)

    sd.wait()

# ================= MAIN LOOP =================
def main():
    print("🚀 Starting AI VTuber...")

    ws = connect_vts()

    while True:
        try:
            audio_file = record_audio(DURATION)

            text = transcribe(audio_file)
            if not text.strip():
                continue

            reply = ask_llm(text)

            emotion = detect_emotion(reply)
            trigger_expression(ws, emotion)

            tts_audio = higgs_tts(reply)

            play_audio_with_vts(ws, tts_audio)

        except Exception as e:
            print("❌ ERROR:", e)
            time.sleep(1)

# ================= RUN =================
if __name__ == "__main__":
    main()