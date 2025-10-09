from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Model(str, Enum):
    """Built-in microWakeWord models."""

    OKAY_NABU = "okay_nabu"
    HEY_JARVIS = "hey_jarvis"
    HEY_MYCROFT = "hey_mycroft"
    ALEXA = "alexa"
