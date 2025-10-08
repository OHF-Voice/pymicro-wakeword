import ctypes
import json
import statistics
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Deque, List, Optional, Union

import numpy as np
from pymicro_features import MicroFrontend

from .const import Model
from .wakeword import TfLiteWakeWord, get_platform

_DIR = Path(__file__).parent
_REPO_DIR = _DIR.parent
_MODULE_LIB_DIR = _DIR / "lib"
_REPO_LIB_DIR = _REPO_DIR / "lib"
_MODELS_DIR = _DIR / "models"

SAMPLES_PER_SECOND = 16000
SAMPLES_PER_CHUNK = 160  # 10ms
BYTES_PER_SAMPLE = 2  # 16-bit
BYTES_PER_CHUNK = SAMPLES_PER_CHUNK * BYTES_PER_SAMPLE
SECONDS_PER_CHUNK = SAMPLES_PER_CHUNK / SAMPLES_PER_SECOND
STRIDE = 3


class MicroWakeWord(TfLiteWakeWord):
    """Tensorflow-based wake word detection.

    Parameters
    ----------
    id: str
        Unique id of wake word
    wake_word: str
        Wake word phrase (.e.g, "Okay Nabu")
    tflite_model: str or Path
        Path to Tensorflow-lite model file (.tflite)
    probability_cutoff: float
        Threshold for detection (0-1)
    sliding_window_size: int
        Number of probabilities to average for detection
    trained_languages: list[str]
        List of languages that this model was trained for
    libtensorflowlite_c_path: str | Path
        Path to tensorflowlite_c shared library

    See: https://github.com/kahrendt/microWakeWord/
    """

    def __init__(
        self,
        id: str,  # pylint: disable=redefined-builtin
        *,
        wake_word: str,
        tflite_model: Union[str, Path],
        probability_cutoff: float,
        sliding_window_size: int,
        trained_languages: List[str],
        libtensorflowlite_c_path: Union[str, Path],
    ) -> None:
        """Initialize wakeword."""
        TfLiteWakeWord.__init__(self, libtensorflowlite_c_path)

        self.id = id
        self.wake_word = wake_word
        self.tflite_model = tflite_model
        self.probability_cutoff = probability_cutoff
        self.sliding_window_size = sliding_window_size
        self.trained_languages = trained_languages

        self.is_active = True

        # Load the model and create interpreter
        self.model_path = str(Path(tflite_model).resolve()).encode("utf-8")
        self._load_model()

        self._features: List[np.ndarray] = []
        self._probabilities: Deque[float] = deque(maxlen=self.sliding_window_size)
        self._audio_buffer = bytes()

    def _load_model(self) -> None:
        self.model = self.lib.TfLiteModelCreateFromFile(self.model_path)
        self.interpreter = self.lib.TfLiteInterpreterCreate(self.model, None)
        self.lib.TfLiteInterpreterAllocateTensors(self.interpreter)

        # Access input and output tensor
        self.input_tensor = self.lib.TfLiteInterpreterGetInputTensor(
            self.interpreter, 0
        )
        self.output_tensor = self.lib.TfLiteInterpreterGetOutputTensor(
            self.interpreter, 0
        )

        # Get quantization parameters
        input_q = self.lib.TfLiteTensorQuantizationParams(self.input_tensor)
        output_q = self.lib.TfLiteTensorQuantizationParams(self.output_tensor)

        self.input_scale, self.input_zero_point = input_q.scale, input_q.zero_point
        self.output_scale, self.output_zero_point = output_q.scale, output_q.zero_point

    @staticmethod
    def from_config(
        config_path: Union[str, Path],
        libtensorflowlite_c_path: Optional[Union[str, Path]] = None,
    ) -> "MicroWakeWord":
        """Load a microWakeWord model from a JSON config file.

        Parameters
        ----------
        config_path: str or Path
            Path to JSON configuration file
        """
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as config_file:
            config = json.load(config_file)

        micro_config = config["micro"]

        if libtensorflowlite_c_path is None:
            # Try module lib dir first (inside wheel)
            libtensorflowlite_c_path = next(
                iter(_MODULE_LIB_DIR.glob("*tensorflowlite_c.*")), None
            )

            if not libtensorflowlite_c_path:
                # Try repo dir
                platform = get_platform()
                if not platform:
                    raise ValueError("Unable to detect platform for tensorflowlite_c")

                lib_dir = _REPO_LIB_DIR / platform
                libtensorflowlite_c_path = next(
                    iter(lib_dir.glob("*tensorflowlite_c.*")), None
                )

            if not libtensorflowlite_c_path:
                raise ValueError("Failed to find tensorflowlite_c library")

        return MicroWakeWord(
            id=Path(config["model"]).stem,
            wake_word=config["wake_word"],
            tflite_model=config_path.parent / config["model"],
            probability_cutoff=micro_config["probability_cutoff"],
            sliding_window_size=micro_config["sliding_window_size"],
            trained_languages=micro_config.get("trained_languages", []),
            libtensorflowlite_c_path=libtensorflowlite_c_path,
        )

    @staticmethod
    def from_builtin(
        model: Model,
        models_dir: Union[str, Path] = _MODELS_DIR,
        libtensorflowlite_c_path: Optional[Union[str, Path]] = None,
    ) -> "MicroWakeWord":
        """Load a builtin microWakeWord model."""
        models_dir = Path(models_dir)

        return MicroWakeWord.from_config(
            models_dir / f"{model.value}.json",
            libtensorflowlite_c_path=libtensorflowlite_c_path,
        )

    def process_streaming(self, features: np.ndarray) -> bool:
        """Return True if wake word is detected.

        Parameters
        ----------
        features: ndarray
            Audio features from MicroWakeWordFeatures
        """
        self._features.append(features)

        if len(self._features) < STRIDE:
            # Not enough windows
            return False

        # Allocate and quantize input data
        quant_features = np.round(
            np.concatenate(self._features, axis=1) / self.input_scale
            + self.input_zero_point
        ).astype(np.uint8)

        # Stride instead of rolling
        self._features.clear()

        # Set tensor
        quant_ptr = quant_features.ctypes.data_as(ctypes.c_void_p)
        self.lib.TfLiteTensorCopyFromBuffer(
            self.input_tensor, quant_ptr, quant_features.nbytes
        )

        # Run inference
        self.lib.TfLiteInterpreterInvoke(self.interpreter)

        # Read output
        output_bytes = self.lib.TfLiteTensorByteSize(self.output_tensor)
        output_data = np.empty(output_bytes, dtype=np.uint8)
        self.lib.TfLiteTensorCopyToBuffer(
            self.output_tensor,
            output_data.ctypes.data_as(ctypes.c_void_p),
            output_bytes,
        )

        # Dequantize output
        result = (
            output_data.astype(np.float32) - self.output_zero_point
        ) * self.output_scale

        self._probabilities.append(result.item())

        if len(self._probabilities) < self.sliding_window_size:
            # Not enough probabilities
            return False

        if statistics.mean(self._probabilities) > self.probability_cutoff:
            return True

        return False

    def reset(self) -> None:
        self._features.clear()
        self._probabilities.clear()

        # Need to reload model to reset intermediary results
        self._load_model()


# -----------------------------------------------------------------------------


class MicroWakeWordFeatures:
    """Audio feature generator for microWakeWord."""

    def __init__(
        self,
    ) -> None:
        self._audio_buffer = bytes()
        self._frontend = MicroFrontend()

    def process_streaming(self, audio_bytes: bytes) -> Iterable[np.ndarray]:
        """Generate audio features from raw audio.

        Audio must be 16Khz 16-bit mono.
        """
        self._audio_buffer += audio_bytes

        if len(self._audio_buffer) < BYTES_PER_CHUNK:
            # Not enough audio to get features
            return

        audio_buffer_idx = 0
        while (audio_buffer_idx + BYTES_PER_CHUNK) <= len(self._audio_buffer):
            # Process chunk
            chunk_bytes = self._audio_buffer[
                audio_buffer_idx : audio_buffer_idx + BYTES_PER_CHUNK
            ]
            frontend_result = self._frontend.process_samples(chunk_bytes)
            audio_buffer_idx += frontend_result.samples_read * BYTES_PER_SAMPLE

            if not frontend_result.features:
                # Not enough audio for a full window
                continue

            yield np.array(frontend_result.features).reshape(
                (1, 1, len(frontend_result.features))
            )

        # Remove processed audio
        self._audio_buffer = self._audio_buffer[audio_buffer_idx:]

    def reset(self) -> None:
        self._frontend.reset()
        self._audio_buffer = bytes()
