# Python microWakeWord

Python library for [microWakeWord](https://github.com/kahrendt/microWakeWord).

Uses a [pre-compiled Tensorflow Lite library](https://github.com/tphakala/tflite_c).


## Install

``` sh
pip3 install pymicro-wakeword
```


## Usage

``` python
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures, Model

mww = MicroWakeWord.from_builtin(Model.OKAY_NABU)
mww_features = MicroWakeWordFeatures()

# Audio must be 16-bit mono at 16Khz
while audio := get_10ms_of_audio():
    assert len(audio) == 160 * 2  # 160 samples
    for features in mww_features.process_streaming(audio):
        if mww.process_streaming(features):
            print("Detected!")
```


## Command-Line

### WAVE files

``` sh
python3 -m pymicro_wakeword --model 'okay_nabu' /path/to/*.wav
```

### Live

``` sh
arecord -r 16000 -c 1 -f S16_LE -t raw | \
  python3 -m pymicro_wakeword --model 'okay_nabu'
```
