"""
A program file of the mod's own (an NSO), for code that has no room in the game's.

A Switch game is several program files that its loader lays out one after another: this game's
`rtld`, `main`, `subsdk0` and `sdk`. A mod folder's `exefs` may add one more, `subsdk1` to
`subsdk9`, and the loader lays it out with the rest. Its code can then be as long as it needs to
be, and the game's own code reaches it with an ordinary call.

What a program file must hold is what the game's own `rtld` asks of each one it finds:

* At the start of its code, the place of the module's header (`MOD0`), which `rtld` stops for good
  without. The game's own files have the header 8 bytes in; so has this.
* Writable memory where the header says, for what `rtld` keeps of the module: here, the zeroed
  memory after the module's variables.
* A table of linking with a table of symbols by their hash, which `rtld`'s search divides by the
  number of rows: so this has one row, empty, and nothing is found in it. Nothing in it is to be
  moved or joined, and there is no function to start, so `rtld` passes it over.

And what the emulator asks: each part of the file at the start of a page of memory, the code
first; the constants beginning with the module's name. Nothing is packed.
"""

from __future__ import annotations

import hashlib
import struct

PAGE = 0x1000
HEADER = 0x100  # the header; the module's name field follows it (one 0 byte, as in the game's), then the parts
MOD0_AT = 8  # where the module's header is, as in the game's own files
VARIABLES = 0x10  # the module's variables: none are used; a few zero bytes so that no part is empty
ZEROED = PAGE - VARIABLES  # its zeroed memory: the rest of that page
DT_NULL, DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_SYMENT = 0, 4, 5, 6, 10, 11


def _align(n: int) -> int:
    return (n + PAGE - 1) // PAGE * PAGE


def build(name: str, code: bytes, at: int) -> bytes:
    """
    A program file whose code is `code`, placed `at` bytes from the module's start (past its header).

    Returns:
        The file's bytes.

    Raises:
        ValueError: If the code does not fit after the header.
    """
    if at % 4 or at < MOD0_AT + 0x1C or len(code) % 4 or not code:
        raise ValueError(f"code of {len(code)} bytes at {at:#x} does not fit after the header")
    label = name.encode("ascii")
    # the constants: the module's name, its table of linking, an empty table of symbols by hash, one
    # empty symbol, an empty name, and an empty table of what unwinds a call
    text_size = at + len(code)
    ro_mem = _align(text_size)
    ro = bytearray(struct.pack("<II", 0, len(label)) + label)
    ro += bytes(-len(ro) % 8)
    dyn_at = ro_mem + len(ro)
    ro += bytes(8 * 6)
    hash_at = ro_mem + len(ro)
    ro += struct.pack("<4I", 1, 1, 0, 0)  # one row, one chain, both empty
    sym_at = ro_mem + len(ro)
    ro += bytes(16)  # symbol 0, the one every table begins with
    str_at = ro_mem + len(ro)
    ro += b"\0"
    ro += bytes(-len(ro) % 4)
    eh_hdr_at = ro_mem + len(ro)
    eh_frame_at = eh_hdr_at + 12
    # version 1; where the frames are, as a signed 4-byte distance from that field; the count; the table's kind
    ro += bytes([1, 0x1B, 0x03, 0x3B]) + struct.pack("<iI", eh_frame_at - (eh_hdr_at + 4), 0)
    eh_end = ro_mem + len(ro)  # (the module header names that table alone, as the game's do)
    ro += bytes(4)  # the frames: none, and the word that ends them
    entries = (DT_HASH, hash_at, DT_STRTAB, str_at, DT_SYMTAB, sym_at, DT_STRSZ, 1, DT_SYMENT, 16, DT_NULL, 0)
    struct.pack_into("<12i", ro, dyn_at - ro_mem, *entries)
    # the variables and the zeroed memory after them
    data_mem = _align(ro_mem + len(ro))
    data = bytes(VARIABLES)
    zeroed_at = data_mem + VARIABLES
    # the code, its first two words "0, and the header 8 bytes in", then the header
    text = bytearray(text_size)
    struct.pack_into("<II", text, 0, 0, MOD0_AT)
    text[MOD0_AT : MOD0_AT + 4] = b"MOD0"
    struct.pack_into(
        "<6i",
        text,
        MOD0_AT + 4,
        dyn_at - MOD0_AT,
        zeroed_at - MOD0_AT,
        zeroed_at + ZEROED - MOD0_AT,
        eh_hdr_at - MOD0_AT,
        eh_end - MOD0_AT,
        zeroed_at - MOD0_AT,
    )
    text[at:] = code
    parts = [bytes(text), bytes(ro), data]
    head = bytearray(HEADER)
    head[0:4] = b"NSO0"
    offset = HEADER + 1
    for k, (part, mem) in enumerate(zip(parts, (0, ro_mem, data_mem), strict=True)):
        struct.pack_into("<III", head, 0x10 + 0x10 * k, offset, mem, len(part))
        struct.pack_into("<I", head, 0x60 + 4 * k, len(part))
        head[0xA0 + 0x20 * k : 0xC0 + 0x20 * k] = hashlib.sha256(part).digest()
        offset += len(part)
    struct.pack_into("<I", head, 0x1C, HEADER)  # the module's name field: one 0 byte
    struct.pack_into("<I", head, 0x2C, 1)
    struct.pack_into("<I", head, 0x3C, ZEROED)
    head[0x40:0x54] = hashlib.sha256(b"".join(parts)).digest()[:20]  # its build id, 20 bytes and 12 of 0, as the game's
    struct.pack_into("<6I", head, 0x88, 0, 0, str_at - ro_mem, 1, sym_at - ro_mem, 16)
    return bytes(head) + b"\0" + b"".join(parts)
