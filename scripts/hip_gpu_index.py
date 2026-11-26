"""Prints the HIP device index whose name contains argv[1]; without an argument, the list. Exit 1 if there is none."""
import ctypes
import os
import sys

dll_dir = os.path.join(sys.prefix, "Lib", "site-packages", "_rocm_sdk_devel", "bin")
os.add_dll_directory(dll_dir)
hip = ctypes.CDLL(os.path.join(dll_dir, "amdhip64_7.dll"))
count = ctypes.c_int()
hip.hipGetDeviceCount(ctypes.byref(count))
want = sys.argv[1].lower() if len(sys.argv) > 1 else ""
for i in range(count.value):
    name = ctypes.create_string_buffer(256)
    hip.hipDeviceGetName(name, 256, i)
    text = name.value.decode(errors="replace")
    if not want:
        print(i, text)
    elif want in text.lower():
        print(i)
        sys.exit(0)
sys.exit(1 if want else 0)
