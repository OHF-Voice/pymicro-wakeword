# Python microWakeWord

Python library for [microWakeWord](https://github.com/kahrendt/microWakeWord).


## Install

``` sh
pip3 install pymicro-wakeword
```


## Usage

``` python
from pymicro_wakeword import MicroWakeWord, Model

mww = MicroWakeWord.from_builtin(Model.OKAY_NABU)

# Audio must be 16-bit mono at 16Khz
while audio := get_10ms_of_audio():
    assert len(audio) == 160 * 2  # 160 samples
    if mww.process_streaming(audio):
        print("Detected!")
```

