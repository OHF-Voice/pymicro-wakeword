# Changelog

## 2.4.1

- Fix 32-bit ARM wheel workflow

## 2.4.0

- Add support for 32-bit ARM (`armv7l` / `armhf`)
- Fix `int8` conversion

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
