"""Verify identical PE .text sections for corrected-baseline and disabled benchmarks.

Usage: python tools/check-disabled-code.py BASELINE_EXE DISABLED_EXE
Build both with the same compiler, flags, and benchmark source.
"""
import hashlib
import pathlib
import struct
import sys


def machine_code(path):
    data = pathlib.Path(path).read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    assert data[pe:pe + 4] == b"PE\0\0", "Expected a Windows PE executable"
    count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    for i in range(count):
        entry = pe + 24 + optional_size + i * 40
        if data[entry:entry + 8].rstrip(b"\0") == b".text":
            size, offset = struct.unpack_from("<II", data, entry + 16)
            return data[offset:offset + size]
    raise ValueError("Executable has no .text section")


baseline, disabled = map(machine_code, sys.argv[1:3])
if baseline != disabled:
    raise SystemExit("Machine-code sections differ; inspect assembly before claiming zero overhead")
print(f"Identical .text: {len(disabled)} bytes; SHA-256 {hashlib.sha256(disabled).hexdigest()}")
