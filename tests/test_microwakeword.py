"""Tests for microWakeWord."""

import itertools
import wave
from pathlib import Path
from typing import Optional, Union

import pytest

from pymicro_wakeword import MicroWakeWord, Model, MicroWakeWordFeatures

_DIR = Path(__file__).parent

_MODELS = set(Model)
_NUM_WAVS = 3


def _load_wav(name: Union[str, Path], number: Optional[int] = None) -> bytes:
    if number is None:
        wav_path = Path(name)
    else:
        wav_path = _DIR / name / f"{number}.wav"

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


def test_model_with_different_stride() -> None:
    """Test a model exported from the training notebook (stride=2)."""
    model_dir = _DIR / "computer"
    mww = MicroWakeWord.from_config(model_dir / "computer.json")
    mww_features = MicroWakeWordFeatures()

    # positive
    audio_bytes = _load_wav(model_dir / "computer.wav")
    detected = False
    for features in mww_features.process_streaming(audio_bytes):
        if mww.process_streaming(features):
            detected = True
            break

    assert detected, "expected computer to be detected"

    mww.reset()
    mww_features.reset()

    # negative
    audio_bytes = _load_wav(model_dir / "hey_peppa_pig.wav")
    detected = False
    for features in mww_features.process_streaming(audio_bytes):
        if mww.process_streaming(features):
            detected = True
            break

    assert not detected, "expected computer NOT to be detected"


def test_close() -> None:
    """Test releasing of resources."""
    mww = MicroWakeWord.from_builtin(Model.OKAY_NABU)
    mww_features = MicroWakeWordFeatures()

    features = list(mww_features.process_streaming(bytes(16000 * 2)))
    assert features

    assert all(mww.process_streaming(f) is False for f in features)

    # Release resources
    mww.close()
    assert all(mww.process_streaming(f) is None for f in features)
