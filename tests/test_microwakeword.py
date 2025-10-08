"""Tests for microWakeWord."""
import itertools
import wave
from pathlib import Path

import pytest

from pymicro_wakeword import MicroWakeWord, Model, MicroWakeWordFeatures

_DIR = Path(__file__).parent

_MODELS = set(Model)
_NUM_WAVS = 3


def _load_wav(model_name: str, number: int) -> bytes:
    wav_path = _DIR / model_name / f"{number}.wav"
    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getsampwidth() == 2
        assert wav_file.getnchannels() == 1

        return wav_file.readframes(wav_file.getnframes())


@pytest.mark.parametrize(
    "model,number", list(itertools.product(_MODELS, range(1, _NUM_WAVS + 1)))
)
def test_process_streaming(model: Model, number: int) -> None:
    """Test streaming processing."""
    mww = MicroWakeWord.from_builtin(model)
    mww_features = MicroWakeWordFeatures()

    # positive
    audio_bytes = _load_wav(model.value, number)
    detected = False
    for features in mww_features.process_streaming(audio_bytes):
        if mww.process_streaming(features):
            detected = True
            break

    assert detected, (model.value, number)

    # Use other wake word samples as negative samples
    for other_model in _MODELS:
        if model == other_model:
            continue

        detected = False
        mww.reset()
        mww_features.reset()

        audio_bytes = _load_wav(other_model.value, number)
        for features in mww_features.process_streaming(audio_bytes):
            if mww.process_streaming(features):
                detected = True
                break

        assert not detected, (model.value, other_model.value, number)
