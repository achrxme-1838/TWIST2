from __future__ import annotations

import ctypes
import os
from typing import Dict, Optional

_EGL_CUDA_DEVICE_NV = 0x323A


def egl_cuda_device_map() -> Dict[int, int]:
    """{EGL device index: CUDA device index} for the NVIDIA EGL devices; {} if EGL
    or the device-query extensions are unavailable."""
    try:
        lib = ctypes.CDLL("libEGL.so.1")
    except OSError:
        return {}
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    def proc(name, restype, *argtypes):
        addr = lib.eglGetProcAddress(name)
        return ctypes.CFUNCTYPE(restype, *argtypes)(addr) if addr else None

    query_devices = proc(b"eglQueryDevicesEXT", ctypes.c_uint,
                         ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    query_attrib = proc(b"eglQueryDeviceAttribEXT", ctypes.c_uint,
                        ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))
    if query_devices is None or query_attrib is None:
        return {}

    n = ctypes.c_int(0)
    if not query_devices(0, None, ctypes.byref(n)) or n.value <= 0:
        return {}
    devs = (ctypes.c_void_p * n.value)()
    if not query_devices(n.value, devs, ctypes.byref(n)):
        return {}

    mapping = {}
    for i in range(n.value):
        cuda = ctypes.c_ssize_t(-1)
        if query_attrib(devs[i], _EGL_CUDA_DEVICE_NV, ctypes.byref(cuda)) and cuda.value >= 0:
            mapping[i] = int(cuda.value)
    return mapping


def select_egl_device_for_cuda(cuda_index: int = 0, verbose: bool = True) -> Optional[int]:
    """Set MUJOCO_EGL_DEVICE_ID to the EGL device on CUDA device ``cuda_index``
    (in the process's visible CUDA numbering; 0 = the GPU CUDA work lands on by
    default). An explicitly set MUJOCO_EGL_DEVICE_ID is respected. Returns the EGL
    index chosen, or None when nothing was changed."""
    if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
        return None
    mapping = egl_cuda_device_map()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    for egl_idx, cuda_idx in mapping.items():
        if cuda_idx == cuda_index:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_idx)
            if verbose:
                print(f"[gl] MUJOCO_EGL_DEVICE_ID={egl_idx} -> CUDA device {cuda_index}"
                      + (f" (CUDA_VISIBLE_DEVICES={visible})" if visible else "")
                      + f"; EGL->CUDA map {mapping}")
            return egl_idx
    if verbose:
        print(f"[gl] no EGL device reports CUDA device {cuda_index} (map={mapping}); "
              f"leaving MUJOCO_EGL_DEVICE_ID unset.")
    return None
