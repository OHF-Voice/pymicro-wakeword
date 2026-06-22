# Changelog

## Unreleased

- Add support for 32-bit ARM (`armv7l` / `armhf`, e.g. Raspberry Pi 2/3 in 32-bit,
  Orange Pi / Allwinner H3). Previously `arm` matched `linux_arm64`, so the 64-bit
  TFLite library was selected on 32-bit ARM, failing at load with
  `wrong ELF class: ELFCLASS64`. Platform detection now distinguishes 32-bit ARM and
  loads `lib/linux_armv7l/libtensorflowlite_c.so`. Adds an `armv7l` wheel build (QEMU).

## 2.3.0

- Add `process_streaming_prob` to get wake word probability

## 2.2.1

- Add a flag for debug logging of wake word probabilities

## 2.2.0

- Detect model stride from input tensor (fix notebook trained models)

## 2.1.0

- Free resources in `__del__`
- Add py.typed

## 2.0.0

- Use embedded TFLite library with ctypes
- Separate out feature processing
- Remove `process_clip`
- Include Alexa model
- Transition to pyproject.toml

## 1.0.0

- Initial release
