"""Tensorflow-based wake word detection."""
from .const import Model
from .microwakeword import MicroWakeWord

__all__ = [
    "MicroWakeWord",
    "Model",
]
