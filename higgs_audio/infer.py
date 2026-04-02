import argparse
import numpy as np
import scipy.io.wavfile as wav

# 🔥 TEMP FAKE until we wire real model
# Replace later with actual Higgs model

def generate_audio(text):
    rate = 16000
    t = np.linspace(0, 1.5, int(rate * 1.5))

    # simple tone based on text length
    freq = 220 + (len(text) * 10)
    audio = 0.3 * np.sin(2 * np.pi * freq * t)

    return (audio * 32767).astype(np.int16), rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    audio, rate = generate_audio(args.text)

    wav.write(args.output, rate, audio)

    print("✅ Generated audio:", args.output)


if __name__ == "__main__":
    main()