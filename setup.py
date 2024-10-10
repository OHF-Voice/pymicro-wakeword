#!/usr/bin/env python3
from pathlib import Path

import setuptools
from setuptools import setup

this_dir = Path(__file__).parent

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

module = "pymicro_wakeword"
module_dir = this_dir / module
models_dir = module_dir / "models"

data_files = list(models_dir.glob("*.json")) + list(models_dir.glob("*.tflite"))

# -----------------------------------------------------------------------------

setup(
    name=module,
    version="1.0.0",
    description="A TensorFlow based wake word detection training framework using synthetic sample generation suitable for certain microcontrollers.",
    url="https://github.com/OHF-Voice/pymicro-wakeword",
    author="Michael Hansen",
    author_email="mike@rhasspy.org",
    license="Apache-2.0",
    packages=setuptools.find_packages(),
    package_data={module: [str(p.relative_to(module_dir)) for p in data_files]},
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="wake word detection hotword",
)
