"""microWakeWord implementation."""
import json
import statistics
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Union

import numpy as np
import tensorflow as tf
from pymicro_features import MicroFrontend

from .const import ClipResult, Model

_DIR = Path(__file__).parent
_MODELS_DIR = _DIR / "models"

_SAMPLES_PER_SECOND = 16000
_SAMPLES_PER_CHUNK = 160  # 10ms
_BYTES_PER_SAMPLE = 2  # 16-bit
_BYTES_PER_CHUNK = _SAMPLES_PER_CHUNK * _BYTES_PER_SAMPLE
_SECONDS_PER_CHUNK = _SAMPLES_PER_CHUNK / _SAMPLES_PER_SECOND
_STRIDE = 3
_DEFAULT_REFRACTORY = 2  # seconds


class MicroWakeWord:
    """Tensorflow-based wake word detection.

    Parameters
    ----------
    wake_word: str
        Wake word phrase (.e.g, "Okay Nabu")
    tflite_model: str or Path
        Path to Tensorflow-lite model file (.tflite)
    probability_cutoff: float
        Threshold for detection (0-1)
    sliding_window_size: int
        Number of probabilities to average for detection
    refractory_seconds: float
        Number of seconds to ignore after detection

    See: https://github.com/kahrendt/microWakeWord/
    """

    def __init__(
        self,
        wake_word: str,
        tflite_model: Union[str, Path],
        probability_cutoff: float,
        sliding_window_size: int,
        refractory_seconds: float,
    ) -> None:
        self.wake_word = wake_word
        self.tflite_model = tflite_model
        self.probability_cutoff = probability_cutoff
        self.sliding_window_size = sliding_window_size
        self.refractory_seconds = refractory_seconds

        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model))
        self.output_details = self.interpreter.get_output_details()[0]
        self.input_details = self.interpreter.get_input_details()[0]
        self.interpreter.allocate_tensors()

        self.data_type = self.input_details["dtype"]
        input_quantization_parameters = self.input_details["quantization_parameters"]
        self.input_scale, self.input_zero_point = (
            input_quantization_parameters["scales"][0],
            input_quantization_parameters["zero_points"][0],
        )

        output_quantization_parameters = self.output_details["quantization_parameters"]
        self.output_scale = output_quantization_parameters["scales"][0]
        self.output_zero_point = output_quantization_parameters["zero_points"][0]

        self._frontend = MicroFrontend()
        self._features: List[np.ndarray] = []
        self._probabilities: Deque[float] = deque(maxlen=self.sliding_window_size)
        self._audio_buffer = bytes()
        self._ignore_seconds: float = 0

    def reset(self) -> None:
        """Reload model and clear state.

        This must be done between audio clips when not streaming.
        """
        self._audio_buffer = bytes()
        self._features.clear()
        self._probabilities.clear()
        self._ignore_seconds = 0

        # Clear out residual features
        self._frontend = MicroFrontend()

        # Need to reload model to reset intermediary results
        # reset_all_variables() doesn't work.
        self.interpreter = tf.lite.Interpreter(model_path=str(self.tflite_model))
        self.interpreter.allocate_tensors()

    @property
    def samples_per_chunk(self) -> int:
        """Number of samples in a streaming audio chunk."""
        return _SAMPLES_PER_CHUNK

    @property
    def bytes_per_chunk(self) -> int:
        """Number of bytes in a streaming audio chunk.

        Assumes 16-bit mono samples at 16Khz.
        """
        return _BYTES_PER_CHUNK

    @staticmethod
    def from_builtin(
        model: Model,
        models_dir: Union[str, Path] = _MODELS_DIR,
        refractory_seconds: float = _DEFAULT_REFRACTORY,
    ) -> "MicroWakeWord":
        """Load a builtin microWakeWord model.

        Parameters
        ----------
        model: Model enum
            Name of builtin model to load
        models_dir: str or Path
            Path to directory where Tensorflow-Lite models and JSON configs exist
        refractory_seconds: float
            Number of seconds to ignore after detection
        """
        config_path = Path(models_dir) / f"{model.value}.json"
        return MicroWakeWord.from_config(
            config_path, refractory_seconds=refractory_seconds
        )

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        refractory_seconds: float = _DEFAULT_REFRACTORY,
    ) -> "MicroWakeWord":
        """Load a microWakeWord model from a JSON config file.

        Parameters
        ----------
        config_path: str or Path
            Path to JSON configuration file
        refractory_seconds: float
            Number of seconds to ignore after detection
        """
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        micro_config = config["micro"]

        return MicroWakeWord(
            wake_word=config["wake_word"],
            tflite_model=config_path.parent / config["model"],
            probability_cutoff=micro_config["probability_cutoff"],
            sliding_window_size=micro_config["sliding_window_size"],
            refractory_seconds=refractory_seconds,
        )

    def process_clip(self, audio_bytes: bytes) -> ClipResult:
        """Process an entire audio clip.

        Parameters
        ----------
        audio_bytes: bytes
            Raw 16-bit mono audio samples at 16Khz
        """
        audio_idx = 0
        num_audio_bytes = len(audio_bytes)
        features: List[List[float]] = []
        while (audio_idx + _BYTES_PER_CHUNK) <= num_audio_bytes:
            frontend_result = self._frontend.ProcessSamples(
                audio_bytes[audio_idx : audio_idx + _BYTES_PER_CHUNK]
            )
            audio_idx += frontend_result.samples_read * _BYTES_PER_SAMPLE
            if frontend_result.features:
                features.append(frontend_result.features)

        if len(features) < _STRIDE:
            # Not enough audio
            self._frontend = MicroFrontend()
            return ClipResult(detected=False)

        # Process all features
        features_array = np.expand_dims(np.array(features), 0)
        quant_features = (features_array / self.input_scale) + self.input_zero_point
        quant_features = quant_features.astype(self.data_type)

        num_windows = quant_features.shape[1]
        probabilities: List[float] = []
        detected_idx: Optional[int] = None
        for features_idx in range(0, num_windows - (num_windows % _STRIDE), _STRIDE):
            self.interpreter.set_tensor(
                self.input_details["index"],
                quant_features[:, features_idx : features_idx + _STRIDE, :],
            )
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details["index"])

            # dequantize output data
            result = self.output_scale * (
                result.astype(np.float32) - self.output_zero_point
            )
            probabilities.append(result.item())

            if (
                (detected_idx is None)
                and (len(probabilities) >= self.sliding_window_size)
                and (
                    statistics.mean(probabilities[-self.sliding_window_size :])
                    > self.probability_cutoff
                )
            ):
                detected_idx = features_idx

        detected_seconds: Optional[float] = None
        if detected_idx is not None:
            audio_seconds = (num_audio_bytes // _BYTES_PER_SAMPLE) / _SAMPLES_PER_SECOND
            detected_seconds = audio_seconds * (detected_idx / num_windows)

        self.reset()

        return ClipResult(
            detected=(detected_idx is not None),
            detected_seconds=detected_seconds,
            probabilities=probabilities,
        )

    def process_streaming(self, audio_bytes: bytes) -> bool:
        """Process a chunk of audio in streaming mode.

        Parameters
        ----------
        audio_bytes: bytes
            Raw 16-bit mono audio samples at 16Khz

        Returns True if the wake word was detected.
        """
        self._audio_buffer += audio_bytes

        if len(self._audio_buffer) < _BYTES_PER_CHUNK:
            # Not enough audio to get features
            return False

        detected = False
        audio_buffer_idx = 0
        while (audio_buffer_idx + _BYTES_PER_CHUNK) <= len(self._audio_buffer):
            # Process chunk
            chunk_bytes = self._audio_buffer[
                audio_buffer_idx : audio_buffer_idx + _BYTES_PER_CHUNK
            ]
            frontend_result = self._frontend.ProcessSamples(chunk_bytes)
            audio_buffer_idx += frontend_result.samples_read * _BYTES_PER_SAMPLE
            self._ignore_seconds = max(0, self._ignore_seconds - _SECONDS_PER_CHUNK)

            if not frontend_result.features:
                # Not enough audio for a full window
                continue

            self._features.append(
                np.array(frontend_result.features).reshape(
                    (1, 1, len(frontend_result.features))
                )
            )

            if len(self._features) < _STRIDE:
                # Not enough windows
                continue

            # quantize the input data
            quant_features = (
                np.concatenate(self._features, axis=1) / self.input_scale
            ) + self.input_zero_point
            quant_features = quant_features.astype(self.data_type)

            # Stride instead of rolling
            self._features.clear()

            self.interpreter.set_tensor(self.input_details["index"], quant_features)
            self.interpreter.invoke()
            result = self.interpreter.get_tensor(self.output_details["index"])

            # dequantize output data
            result = self.output_scale * (
                result.astype(np.float32) - self.output_zero_point
            )
            self._probabilities.append(result.item())

            if len(self._probabilities) < self.sliding_window_size:
                # Not enough probabilities
                continue

            if statistics.mean(self._probabilities) > self.probability_cutoff:
                if self._ignore_seconds <= 0:
                    detected = True
                    self._ignore_seconds = self.refractory_seconds

        # Remove processed audio
        self._audio_buffer = self._audio_buffer[audio_buffer_idx:]

        return detected
