"""Base class for wake words."""

import ctypes as C
import os
import platform
import sys
from pathlib import Path
from typing import Optional, Union

TfLiteStatus = C.c_int  # kTfLiteOk == 0


class TfLiteQuantizationParams(C.Structure):
    _fields_ = [("scale", C.c_float), ("zero_point", C.c_int32)]


c_void_p = C.c_void_p
c_int32 = C.c_int32
c_size_t = C.c_size_t


def _not_null(result, func, args):
    if not result:
        raise RuntimeError(f"{func.__name__} returned NULL")
    return result


class TfLiteWakeWord:
    def __init__(self, libtensorflowlite_c_path: Union[str, Path]):
        self.libtensorflowlite_c_path = Path(libtensorflowlite_c_path).resolve()

        if sys.platform == "win32":
            self.lib = C.CDLL(str(self.libtensorflowlite_c_path))
        else:
            # Load ONCE with RTLD_GLOBAL
            self.lib = C.CDLL(str(self.libtensorflowlite_c_path), mode=os.RTLD_GLOBAL)

        lib = self.lib

        # Model / Interpreter
        lib.TfLiteModelCreateFromFile.argtypes = [C.c_char_p]
        lib.TfLiteModelCreateFromFile.restype = c_void_p

        lib.TfLiteInterpreterCreate.argtypes = [c_void_p, c_void_p]
        lib.TfLiteInterpreterCreate.restype = c_void_p

        lib.TfLiteInterpreterAllocateTensors.argtypes = [c_void_p]
        lib.TfLiteInterpreterAllocateTensors.restype = TfLiteStatus

        lib.TfLiteInterpreterInvoke.argtypes = [c_void_p]
        lib.TfLiteInterpreterInvoke.restype = TfLiteStatus

        # Tensors: get
        lib.TfLiteInterpreterGetInputTensor.argtypes = [c_void_p, c_int32]
        lib.TfLiteInterpreterGetInputTensor.restype = c_void_p
        lib.TfLiteInterpreterGetInputTensor.errcheck = _not_null

        lib.TfLiteInterpreterGetOutputTensor.argtypes = [c_void_p, c_int32]
        lib.TfLiteInterpreterGetOutputTensor.restype = c_void_p
        lib.TfLiteInterpreterGetOutputTensor.errcheck = _not_null

        # Sizes / dims
        lib.TfLiteTensorByteSize.argtypes = [c_void_p]
        lib.TfLiteTensorByteSize.restype = c_size_t

        lib.TfLiteTensorNumDims.argtypes = [c_void_p]
        lib.TfLiteTensorNumDims.restype = c_int32

        lib.TfLiteTensorDim.argtypes = [c_void_p, c_int32]
        lib.TfLiteTensorDim.restype = c_int32

        # Type & quant params
        lib.TfLiteTensorType.argtypes = [c_void_p]
        lib.TfLiteTensorType.restype = c_int32

        lib.TfLiteTensorQuantizationParams.argtypes = [c_void_p]
        lib.TfLiteTensorQuantizationParams.restype = TfLiteQuantizationParams

        # Resize
        lib.TfLiteInterpreterResizeInputTensor.argtypes = [
            c_void_p,
            c_int32,
            C.POINTER(c_int32),
            c_int32,
        ]
        lib.TfLiteInterpreterResizeInputTensor.restype = TfLiteStatus

        # Copy buffers
        lib.TfLiteTensorCopyFromBuffer.argtypes = [c_void_p, c_void_p, c_size_t]
        lib.TfLiteTensorCopyFromBuffer.restype = TfLiteStatus

        lib.TfLiteTensorCopyToBuffer.argtypes = [c_void_p, c_void_p, c_size_t]
        lib.TfLiteTensorCopyToBuffer.restype = TfLiteStatus


def get_platform() -> Optional[str]:
    machine = platform.machine().lower()
    is_arm = ("arm" in machine) or ("aarch" in machine)
    is_amd64 = machine in ("x86_64", "amd64")
    system = sys.platform

    if system.startswith("linux"):
        if is_arm:
            return "linux_arm64"

        if is_amd64:
            return "linux_amd64"

    if system == "win32":
        if is_amd64:
            return "windows_amd64"

    if system == "darwin":
        if is_arm:
            return "darwin_arm64"

    # Not supported
    return None
