"""
Reading numbers out of the game's MIPS code without running it.

Ocarina of Time keeps many of the numbers this mod needs inside code rather than data: how fast his
roll plays, how long a swing is held before a charge, how the ring of his spin grows. They are read
here the way a person reads a listing: straight down, following which register holds what. A
routine is taken to start where it makes room on the stack (`addiu sp, sp, -n`).

Words are the code's 32-bit instructions, big-endian, as `struct.unpack` gives them.
"""

from __future__ import annotations

import struct

from itertools import pairwise

Stores = list[tuple[int, int, int, int]]

KEPT = (16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30)  # the registers a call leaves as they were


def signed(word: int) -> int:
    """The low half of an instruction as a signed number."""
    return (word & 0xFFFF ^ 0x8000) - 0x8000


def words_of(code: bytes) -> tuple[int, ...]:
    """A file of code as its instructions."""
    return struct.unpack_from(f">{len(code) // 4}I", code, 0)


def to_float(bits: int) -> float:
    """A word's bits as a single-precision float."""
    return struct.unpack(">f", struct.pack(">I", bits))[0]


def starts(words: tuple[int, ...]) -> list[int]:
    """Where every routine starts: each place that makes room on the stack."""
    return [k for k, w in enumerate(words) if w >> 16 == 0x27BD and w & 0x8000]


def routines(words: tuple[int, ...]) -> list[tuple[int, int]]:
    """(first, end) of every routine, in order."""
    found = starts(words)
    return list(zip(found, [*found[1:], len(words)], strict=True))


def routine(words: tuple[int, ...], i: int) -> tuple[int, int]:
    """(first, end) of the routine that instruction `i` is in."""
    found = starts(words)
    return max([k for k in found if k <= i] or [0]), min([k for k in found if k > i] or [len(words)])


def constants(code: bytes) -> tuple[Stores, set[int]]:
    """
    The constants a file of code builds, and where it stores them.

    A register holds a constant from where one is put in it (the upper half, then perhaps a lower
    half added or or-ed in, or a copy of another's) to where anything else writes it, a call
    included; a routine starts with none. That is how the game's code makes the words of a display
    list, which it stores two at a time. A constant set on one side of a branch is taken as read on
    the other, so what comes of this is checked by what it leads to.

    Returns:
        [(instruction, register stored through, offset, the word stored)], and every constant built.
    """
    words = words_of(code)
    regs, stores, built, call = [0] + [None] * 31, [], set(), -2
    for i, w in enumerate(words):
        op, rs, rt, rd, imm = w >> 26, w >> 21 & 31, w >> 16 & 31, w >> 11 & 31, w & 0xFFFF
        if w >> 16 == 0x27BD and imm & 0x8000:  # room made on the stack: a routine begins
            regs = [0] + [None] * 31
        if op == 0x0F:  # the upper half
            regs[rt] = imm << 16
            built.add(regs[rt])
        elif op in {0x09, 0x0D}:  # a number added, or or-ed in
            v = regs[rs]
            if v is not None:
                v = (v + (imm ^ 0x8000) - 0x8000) & 0xFFFFFFFF if op == 0x09 else v | imm
                built.add(v)
            regs[rt] = v
        elif op == 0x2B:  # a word stored
            if regs[rt] is not None:
                stores.append((i, rs, (imm ^ 0x8000) - 0x8000, regs[rt]))
        elif op == 0:
            if w & 0x3F in {0x21, 0x25} and not (rs and rt):
                regs[rd] = regs[rs or rt]  # a copy
            else:
                regs[rd] = None
                if w & 0x3F == 0x09:
                    call = i
        elif op == 0x03:
            call = i
        elif 0x08 <= op <= 0x0E or 0x20 <= op <= 0x27 or (op in {0x10, 0x11} and rs in {0, 2}):
            regs[rt] = None  # other arithmetic, a load, a move from a coprocessor
        regs[0] = 0
        if call == i - 1:  # the instruction after a call runs before it
            regs = [v if r in KEPT or not r else None for r, v in enumerate(regs)]
    return stores, built


def placed_commands(stores: Stores) -> list[tuple[int, tuple[int, int]]]:
    """
    The display-list commands among the words `constants` found stored.

    A command is two constants stored through one register, at 0 and at 4 from it, as the game's
    code stores one. A command with a word that is not a constant is not among them.

    Returns:
        [(the instruction that stores its first half, (first word, second word))].
    """
    out = []
    for (i, base, off, v), (j, base2, off2, v2) in pairwise(stores):
        if base == base2 and j - i <= 2 and {off, off2} == {0, 4}:
            out.append((min(i, j), (v, v2) if off == 0 else (v2, v)))
    return out


def commands(stores: Stores) -> list[tuple[int, int]]:
    """The commands of `placed_commands`, without where they were stored."""
    return [command for _, command in placed_commands(stores)]


def writes(w: int, reg: int) -> bool:
    """Whether an instruction puts something in register `reg` (of the processor's own 32)."""
    op = w >> 26
    if op == 0:
        return (w >> 11 & 31) == reg and w & 0x3F not in {0x08, 0x18, 0x19, 0x1A, 0x1B}  # (not jr, mult, div)
    if 0x08 <= op <= 0x0F or 0x20 <= op <= 0x27:
        return (w >> 16 & 31) == reg
    if op == 0x11 and w >> 21 & 31 in {0x00, 0x02}:  # mfc1, cfc1
        return (w >> 16 & 31) == reg
    return op == 0x03 and reg == 31


def value(words: tuple[int, ...], i: int, reg: int, floor: int) -> int | None:
    """
    The number in register `reg` as instruction `i` runs, read back as far as `floor`.

    Returns:
        The number, if the instructions before build it (an upper half with a lower half or-ed or
        added in, a number added to or or-ed with nothing, or nothing); None if the last write to
        the register is anything else.
    """
    if reg == 0:
        return 0
    low = None
    for j in range(i - 1, floor - 1, -1):
        w = words[j]
        if not writes(w, reg):
            continue
        op, rs = w >> 26, w >> 21 & 31
        if op in {0x09, 0x0D} and rs == reg and low is None:  # addiu / ori into itself: on to the upper half
            low = (op, w & 0xFFFF)
            continue
        if op == 0x0F:
            v = (w & 0xFFFF) << 16
        elif op in {0x09, 0x0D} and rs == 0:
            v = signed(w) & 0xFFFFFFFF if op == 0x09 else w & 0xFFFF
        elif op == 0 and w & 0x3F in {0x21, 0x25} and not (rs or w >> 16 & 31):
            v = 0  # move reg, zero
        else:
            return None
        if low:
            v = (v + signed(low[1])) & 0xFFFFFFFF if low[0] == 0x09 else v | low[1]
        return v
    return None


def traced(words: tuple[int, ...], i: int, reg: int, floor: int, depth: int = 6) -> int | None:
    """
    As `value`, following copies, sums and or-ings from other registers as well.

    Returns:
        The number in `reg` as instruction `i` runs, if the instructions back to `floor` build it
        from constants alone; None if they do not.
    """
    if reg == 0:
        return 0
    if depth < 0:
        return None
    for j in range(i - 1, floor - 1, -1):
        w = words[j]
        if not writes(w, reg):
            continue
        op, rs, rt = w >> 26, w >> 21 & 31, w >> 16 & 31
        if op == 0x0F:
            return (w & 0xFFFF) << 16
        if op in {0x09, 0x0D}:
            v = traced(words, j, rs, floor, depth - 1)
            if v is None:
                return None
            return (v + signed(w)) & 0xFFFFFFFF if op == 0x09 else v | w & 0xFFFF
        if op == 0 and w & 0x3F in {0x21, 0x25} and 0 in {rs, rt}:
            return traced(words, j, rs or rt, floor, depth - 1)
        return None
    return None


def handed(words: tuple[int, ...], i: int, reg: int, floor: int) -> int | None:
    """The number a call at `i` is handed in `reg`, as `traced` reads it (the instruction after a call runs first)."""
    seq = [*words[floor:i], words[i + 1]]
    return traced(seq, len(seq), reg, 0)


def float_value(words: tuple[int, ...], i: int, freg: int, floor: int, data: bytes, ram: int) -> float | None:
    """
    The number in float register `freg` as instruction `i` runs, read back as far as `floor`.

    Returns:
        The number, if it is loaded from `data` (a file of code sitting at `ram`) or moved across
        from a register `value` can read; None otherwise.
    """
    for j in range(i - 1, floor - 1, -1):
        w = words[j]
        op = w >> 26
        if op == 0x31 and (w >> 16 & 31) == freg:  # lwc1
            base = value(words, j, w >> 21 & 31, floor)
            if base is None:
                return None
            at = ((base + signed(w)) & 0xFFFFFFFF) - ram
            return struct.unpack(">f", data[at : at + 4])[0] if 0 <= at <= len(data) - 4 else None
        if op == 0x11 and w >> 21 & 31 == 0x04 and (w >> 11 & 31) == freg:  # mtc1
            v = value(words, j, w >> 16 & 31, floor)
            return None if v is None else to_float(v)
        if op == 0x11 and w >> 21 & 31 in {0x10, 0x11, 0x14} and (w >> 6 & 31) == freg:
            return None  # worked out: not a number kept
    return None


def origin(words: tuple[int, ...], i: int, reg: int, floor: int) -> int | None:
    """Where `value` finds the number in `reg` as instruction `i` runs: the instruction that sets its upper half."""
    for j in range(i - 1, floor - 1, -1):
        w = words[j]
        if not writes(w, reg):
            continue
        op, rs = w >> 26, w >> 21 & 31
        if op in {0x09, 0x0D} and rs == reg:
            continue  # (the lower half: on to the upper)
        return j if op == 0x0F or (op in {0x09, 0x0D} and rs == 0) else None
    return None


def calls(words: tuple[int, ...], first: int, end: int, low: int) -> list[tuple[int, int]]:
    """[(instruction, target)] for every call in a stretch of a file of code sitting at `low`."""
    return [(i, (words[i] & 0x3FFFFFF) << 2 | low & 0xF0000000) for i in range(first, end) if words[i] >> 26 == 3]


def argument(words: tuple[int, ...], i: int, reg: int, floor: int) -> int | None:
    """The number a call at `i` is handed in `reg`, read back to `floor` (the instruction after a call runs first)."""
    seq = [*words[floor:i], words[i + 1]]
    return value(seq, len(seq), reg, 0)


def loads(words: tuple[int, ...], first: int, last: int, ops: tuple[int, ...] = (0x23,)) -> list[tuple[int, int]]:
    """
    Every load from an address between `first` and `last`: an upper half put in a register, and a
    load through that register within the next few instructions.

    Returns:
        [(the instruction that loads, the address)].
    """
    out = []
    for i, w in enumerate(words):
        if w >> 26 != 0x0F:
            continue
        reg = w >> 16 & 31
        for j in range(i + 1, min(i + 10, len(words))):
            v = words[j]
            if v >> 26 == 0x0F and v >> 16 & 31 == reg:
                break
            if v >> 26 in ops and v >> 21 & 31 == reg:
                address = ((w & 0xFFFF) << 16) + ((v & 0xFFFF ^ 0x8000) - 0x8000)
                if first <= address <= last:
                    out.append((j, address))
                break
    return out


def floats(words: tuple[int, ...], first: int, end: int) -> dict[int, tuple[int, float]]:
    """
    The numbers a stretch builds in a float register: an upper half put in the assembler's register
    and moved across.

    Returns:
        {instruction: (float register, number)}.
    """
    return {
        i: (words[i + 1] >> 11 & 31, to_float((words[i] & 0xFFFF) << 16))
        for i in range(first, end - 1)
        if words[i] >> 16 == 0x3C01 and words[i + 1] & 0xFFFF07FF == 0x44810000
    }
