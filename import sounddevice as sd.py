import sounddevice as sd

print("Available audio devices:\n")
for idx, dev in enumerate(sd.query_devices()):
    print(f"{idx}: {dev['name']} (Input: {dev['max_input_channels']}, Output: {dev['max_output_channels']})")