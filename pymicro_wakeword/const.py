from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Model(str, Enum):
    """Built-in microWakeWord models."""

    OKAY_NABU = "okay_nabu"
    HEY_JARVIS = "hey_jarvis"
    HEY_MYCROFT = "hey_mycroft"


@dataclass
class ClipResult:
    """Result of process_clip."""

    detected: bool
    """True if wake word was detected in clip."""

    detected_seconds: Optional[float] = None
    """Seconds into the audio that wake word was detected."""

    probabilities: Optional[List[float]] = None
    """Probabilities for each feature window."""
