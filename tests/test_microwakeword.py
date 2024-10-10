"""Tests for microWakeWord."""
import itertools
import wave
from pathlib import Path

import pytest

from pymicro_wakeword import MicroWakeWord, Model

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
def test_process_clip(model: Model, number: int) -> None:
    """Test full clip processing."""
    mww = MicroWakeWord.from_builtin(model)

    # positive
    audio_bytes = _load_wav(model.value, number)
    assert mww.process_clip(audio_bytes).detected, (model.value, number)

    # Use other wake word samples as negative samples
    for other_model in _MODELS:
        if model == other_model:
            continue

        audio_bytes = _load_wav(other_model.value, number)
        assert not mww.process_clip(audio_bytes).detected, (
            model.value,
            other_model.value,
            number,
        )


@pytest.mark.parametrize(
    "model,number", list(itertools.product(_MODELS, range(1, _NUM_WAVS + 1)))
)
def test_process_streaming(model: Model, number: int) -> None:
    """Test streaming processing."""
    mww = MicroWakeWord.from_builtin(model)

    # positive
    audio_bytes = _load_wav(model.value, number)
    assert mww.process_streaming(audio_bytes), (model.value, number)

    # Use other wake word samples as negative samples
    for other_model in _MODELS:
        if model == other_model:
            continue

        # Reset between uses
        mww.reset()

        audio_bytes = _load_wav(other_model.value, number)
        assert not mww.process_streaming(audio_bytes), (
            model.value,
            other_model.value,
            number,
        )
