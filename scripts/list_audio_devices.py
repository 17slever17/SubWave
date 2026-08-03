import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from services.stt.audio import describe_audio_devices


def main():
    lines = describe_audio_devices()
    if not lines:
        print("No audio devices found.")
        return
    print("Available audio devices:")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
