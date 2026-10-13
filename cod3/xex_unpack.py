"""Unpack an Xbox 360 XEX2 executable (retail AES encryption, basic compression) to its memory image.

Reads the user's own default.xex, writes the PE image as it sits in memory (load address in the
header, 0x82000000 for Call of Duty 3) so code can be disassembled at its real addresses
(the sub_8XXXXXXX names the CoD3Recomp notes use).

Usage: XEX_RETAIL_KEY=<32 hex digits> python -I xex_unpack.py <default.xex> <out image.bin>
Only "basic" compression (data blocks + zero runs) is handled; LZX-compressed XEXs are rejected.
"""
import os
import struct
import sys

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# The Xbox 360 retail XEX key is not included: set XEX_RETAIL_KEY to it (32 hex digits). It is the same
# key emulators such as Xenia use to load retail titles.
RETAIL_KEY = bytes.fromhex(os.environ.get("XEX_RETAIL_KEY", "")) or None


def unpack(xex: bytes):
    magic, _, pe_off, _, sec_off, nopt = struct.unpack_from(">4sIIIII", xex, 0)
    if magic != b"XEX2":
        raise ValueError("not an XEX2 file")
    fmt = None
    for i in range(nopt):
        key, val = struct.unpack_from(">II", xex, 24 + i * 8)
        if key == 0x3FF:
            fmt = val
    size, enc, comp = struct.unpack_from(">IHH", xex, fmt)
    if comp != 1:
        raise ValueError(f"compression type {comp} not supported (only basic)")
    image_size, load_address = struct.unpack_from(">I", xex, sec_off + 4)[0], struct.unpack_from(">I", xex, sec_off + 0x110)[0]
    payload = xex[pe_off:]
    if enc == 1:
        file_key = xex[sec_off + 0x150:sec_off + 0x160]
        if RETAIL_KEY is None:
            sys.exit("set XEX_RETAIL_KEY (the Xbox 360 retail XEX key) to decrypt a retail XEX")
        dec = Cipher(algorithms.AES(RETAIL_KEY), modes.ECB()).decryptor()
        session = dec.update(file_key) + dec.finalize()
        dec = Cipher(algorithms.AES(session), modes.CBC(bytes(16))).decryptor()
        n = len(payload) // 16 * 16
        payload = dec.update(payload[:n]) + dec.finalize()
    out, pos = bytearray(), 0
    for b in range((size - 8) // 8):
        data_size, zero_size = struct.unpack_from(">II", xex, fmt + 8 + b * 8)
        out += payload[pos:pos + data_size]
        out += bytes(zero_size)
        pos += data_size
    if out[:2] != b"MZ":
        raise ValueError("decryption failed (no PE header) - not a retail key?")
    return bytes(out[:image_size]), load_address


if __name__ == "__main__":
    image, base = unpack(open(sys.argv[1], "rb").read())
    open(sys.argv[2], "wb").write(image)
    print(f"image {len(image):#x} bytes at {base:#x}")
