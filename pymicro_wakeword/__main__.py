"""Command-line utility for microWakeWord."""

import argparse
import sys
import wave

from .const import Model
from .microwakeword import MicroWakeWord, MicroWakeWordFeatures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[m.value for m in Model])
    parser.add_argument("--config", help="Path to JSON config")
    parser.add_argument("wav_file", nargs="*")
    args = parser.parse_args()

    if not (args.model or args.config):
        print("--model or --config is required", file=sys.stderr)
        sys.exit(1)

    if args.config:
        mww = MicroWakeWord.from_config(args.config)
    else:
        mww = MicroWakeWord.from_builtin(Model(args.model))

    mww_features = MicroWakeWordFeatures()

    if args.wav_file:
        for wav_path in args.wav_file:
            with wave.open(wav_path, "rb") as wav_file:
                assert wav_file.getframerate() == 16000, "16Khz required"
                assert wav_file.getsampwidth() == 2, "16-bit samples required"
                assert wav_file.getnchannels() == 1, "Mono required"

                audio_bytes = wav_file.readframes(wav_file.getnframes())
                detected = False
                for features in mww_features.process_streaming(audio_bytes):
                    if mww.process_streaming(features):
                        print(wav_path, "detected")
                        detected = True
                        break

                if not detected:
                    print(wav_path, "not-detected")

                mww.reset()
                mww_features.reset()
    else:
        # Live
        try:
            while True:
                chunk = sys.stdin.buffer.read(2048)
                if not chunk:
                    break

                for features in mww_features.process_streaming(chunk):
                    if mww.process_streaming(features):
                        print(args.model, flush=True)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
