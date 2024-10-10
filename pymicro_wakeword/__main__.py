"""Command-line utility for microWakeWord."""
import argparse
import sys
import wave

from pymicro_wakeword import MicroWakeWord, Model


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

    if args.wav_file:
        for wav_path in args.wav_file:
            with wave.open(wav_path, "rb") as wav_file:
                assert wav_file.getframerate() == 16000, "16Khz required"
                assert wav_file.getsampwidth() == 2, "16-bit samples required"
                assert wav_file.getnchannels() == 1, "Mono required"

                audio_bytes = wav_file.readframes(wav_file.getnframes())
                result = mww.process_clip(audio_bytes)
                if result.detected:
                    print(wav_path, "detected", result.detected_seconds)
                else:
                    print(wav_path, "not-detected")

                mww.reset()
    else:
        # Live
        try:
            while True:
                chunk = sys.stdin.buffer.read(mww.bytes_per_chunk)
                if not chunk:
                    break

                if mww.process_streaming(chunk):
                    print(args.model, flush=True)

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
