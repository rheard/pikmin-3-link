"""
An Ocarina of Time ROM: its files, the objects and actors the program's tables name, and the one
picture of Link's head the game keeps.

Nothing from the ROM is kept anywhere: your own copy is read each time. Any byte order is accepted
(.z64, .v64, .n64), and nothing is looked up by a fixed address. The file table is found by its
first entry, the program by the tables it holds, and everything else through those, so other
releases should work, though only the first US one has been read.
"""

from __future__ import annotations

import struct

from pathlib import Path
from typing import NamedTuple

from link.errors import BuildError
from link.formats import yaz0
from link.oot import mips
from link.oot.graphics import Rows, decode_texture

KEEP = 0x01  # the object every scene has loaded: Link's animations' headers, shared textures, the bomb


class Actor(NamedTuple):
    """An actor's code and its record in that code, as the table of actors names them."""

    file: int  # the file of its code
    code: bytes
    ram: int  # where that code sits in memory
    number: int
    held_in: int  # the object it is drawn from
    init: int  # where its routines are, in memory: the one that sets it up ...
    draw: int  # ... and the one that draws it


class Rom:
    """
    The cartridge image, in the console's byte order whichever order the file is in.

    `files` is the file table: (where a file goes, where that ends, where it is in the ROM, where
    that ends or 0 if it is stored as it is). `program` is the game's main file of code, with its
    table of objects at `object_table`; `link_objects` are the adult's and the child's objects.
    """

    def __init__(self, path: str | Path) -> None:
        data = bytearray(Path(path).read_bytes())
        if data[:4] == b"\x37\x80\x40\x12":  # byte-swapped (.v64, and many files named .n64)
            data[0::2], data[1::2] = data[1::2], data[0::2]
        elif data[:4] == b"\x40\x12\x37\x80":  # little-endian words
            for i in range(0, len(data), 4):
                data[i : i + 4] = data[i : i + 4][::-1]
        elif data[:4] != b"\x80\x37\x12\x40":
            raise BuildError(f"{path} is not a Nintendo 64 ROM")
        self.d = bytes(data)
        self.title = self.d[0x20:0x34].decode("ascii", "replace").strip()
        if "ZELDA" not in self.title:
            raise BuildError(f'{path} is "{self.title}", not The Legend of Zelda: Ocarina of Time')
        at = self.d.find(struct.pack(">4I", 0, 0x1060, 0, 0))  # the first entry: the ROM's own header
        if at < 0:
            raise BuildError("the file table was not found in this ROM")
        self.files = []
        while True:
            entry = struct.unpack_from(">4I", self.d, at)
            if not entry[1]:
                break
            self.files.append(entry)
            at += 16
        self.by_start = {entry[0]: i for i, entry in enumerate(self.files)}
        self.found = {}  # what readers of this ROM have found in it, kept so that each thing is found once
        self._cache = {}
        self._actor_table = None
        self._find_objects()

    def file(self, i: int) -> bytes:
        """
        File number `i`, unpacked.

        Returns:
            Its bytes.

        Raises:
            BuildError: If it is packed some other way than Yaz0.
        """
        if i not in self._cache:
            start, end, at, at_end = self.files[i]
            if at_end:
                raw = self.d[at:at_end]
                if raw[:4] != b"Yaz0":
                    raise BuildError(f"file {i} of the ROM is packed in a way this reader does not know")
                self._cache[i] = yaz0.decompress(raw)
            else:
                self._cache[i] = self.d[at : at + end - start]
        return self._cache[i]

    def packed_otherwise(self, i: int) -> bool:
        """Whether file `i` is packed some way that `file` cannot unpack."""
        _, _, at, at_end = self.files[i]
        return bool(at_end) and self.d[at : at + 4] != b"Yaz0"

    def _find_objects(self) -> None:
        """
        Find the program and its table of objects: pairs of a file's start and end.

        The program is the one file holding a long run of such pairs. A run is found from the pair
        of some file in it and walked back to its first entry, which is empty. The program keeps
        three numbers straight before the table, and they are read from there, not assumed: the
        object numbers of the two Links (adult, then child), and how many entries the table has.

        Raises:
            BuildError: If no file holds such a table.
        """
        pairs = {entry[:2] for entry in self.files}
        for i, (start, end, _, at_end) in enumerate(self.files):
            if end - start < 0x80000 or (not at_end and i < 8):
                continue  # too small for the program, or the sound and animation data
            data = self.file(i)
            for probe in self.files[len(self.files) // 3 :: 37]:
                p = data.find(struct.pack(">2I", *probe[:2]))
                while p >= 0:
                    lo = p
                    while (
                        p % 4 == 0
                        and lo >= 16
                        and (struct.unpack_from(">2I", data, lo - 8) in pairs or data[lo - 8 : lo] == bytes(8))
                    ):
                        lo -= 8
                    adult, child, count = struct.unpack_from(">2HI", data, lo - 8)
                    entries = (
                        [struct.unpack_from(">2I", data, lo + 8 * k) for k in range(count)]
                        if 0x40 <= count <= 0x1000
                        else []
                    )
                    # an entry is a file, or empty: nothing, or a start with no length
                    if (
                        entries
                        and all(e in pairs or e[0] == e[1] for e in entries)
                        and 0 < adult < count
                        and 0 < child < count
                    ):
                        self.program_file, self.program = i, data
                        self.object_table, self.object_count = lo, count
                        self.link_objects = (adult, child)
                        return
                    p = data.find(struct.pack(">2I", *probe[:2]), p + 4)
        raise BuildError("the program's table of objects was not found in this ROM")

    def object_file(self, number: int) -> int:
        """
        The file of object `number`.

        Returns:
            Its number in the file table.

        Raises:
            BuildError: If the table has no such object, or it is not a file of the ROM.
        """
        if not 0 < number < self.object_count:
            raise BuildError(f"the ROM has no object {number:#x}")
        start, _ = struct.unpack_from(">2I", self.program, self.object_table + number * 8)
        if start not in self.by_start:
            raise BuildError(f"object {number:#x} is not a file of this ROM")
        return self.by_start[start]

    def object(self, number: int) -> bytes:
        """Object `number`, unpacked."""
        return self.file(self.object_file(number))

    def actor_table(self) -> int:
        """
        Where the game's table of actors is in the program.

        It is the longest run of 32-byte rows that begin with a file's start and end, or with
        nothing. A row says where an actor's code is in the ROM, where that sits in memory, and
        where in it the actor's own record is; the record begins with the actor's number, which
        says how far into the table the row is.

        Returns:
            Where its first row, actor 0's, is in the program.

        Raises:
            BuildError: If no such run is found.
        """
        if self._actor_table is None:
            prog = self.program
            starts = {f[0]: i for i, f in enumerate(self.files) if f[0]}
            best, i = (0, 0), 0
            while i < len(prog) - 0x20:
                a, b = struct.unpack_from(">II", prog, i)
                if a in starts and self.files[starts[a]][1] == b:
                    j = i
                    while j < len(prog) - 0x20:
                        a, b = struct.unpack_from(">II", prog, j)
                        if not ((a in starts and self.files[starts[a]][1] == b) or (a == 0 and b == 0)):
                            break
                        j += 0x20
                    if (j - i) // 0x20 > best[1]:
                        best = (i, (j - i) // 0x20)
                    i = j
                else:
                    i += 4
            first, rows = best
            if rows < 100:
                raise BuildError("the game's table of actors was not found")
            self._actor_table = first - 0x20 * self.actor_at(first).number
        return self._actor_table

    def actor(self, number: int) -> Actor:
        """Actor `number`, by its row of the table of actors."""
        return self.actor_at(self.actor_table() + 0x20 * number)

    def actor_at(self, row: int) -> Actor:
        """
        The actor whose row of the table of actors is at `row` of the program.

        Returns:
            Its code and record.

        Raises:
            BuildError: If the row names no code of its own.
        """
        start, _, ram, ram_end, _, record = struct.unpack_from(">6I", self.program, row)
        if not start or start not in self.by_start or not ram <= record < ram_end:
            raise BuildError(f"the row at {row:#x} of the table of actors has no code of its own")
        code = self.file(self.by_start[start])
        number, _, _, held_in, _, init, _, _, draw = struct.unpack_from(">hBxIhxxI4I", code, record - ram)
        return Actor(self.by_start[start], code, ram, number, held_in, init, draw)

    def map_head(self) -> Rows:
        """
        The small head of Link that marks where he is on a dungeon's map, in the pause screen.

        It is the one picture of his head his game keeps: 16 by 16 on the first release, facing
        left. It is found by what the game does with it. The pause screen's code names a
        full-colour picture, loads it and says how large it is, as three of the commands it stores
        into a display list. Two pictures are drawn that way out of one segment, the markers of the
        dungeon's map: his head, and a skull for the boss. The segment is a file that code loads,
        so it is one of the files the code names by where they are in the ROM: the one in which
        both are pictures (`_marker`), where in every other the same bytes are the middle of
        something else. His is the one with the green cap (`_capped`).

        Returns:
            Rows of RGBA.

        Raises:
            BuildError: If not exactly one such picture is found.
        """
        found = []
        for i in range(len(self.files)):
            if self.packed_otherwise(i):
                continue
            code = self.file(i)
            p = code.find(b"\xfd\x10")  # "the upper half of a register is 0xFD10": a 16-bit picture is named
            while p >= 0 and not (p % 4 == 2 and code[p - 2] == 0x3C):
                p = code.find(b"\xfd\x10", p + 1)
            if p < 0:
                continue
            stores, built = mips.constants(code)
            drawn = {}  # segment -> [(offset, width, height)], as the code draws them
            listed = mips.commands(stores)
            for k, (w0, w1) in enumerate(listed):
                if w0 != 0xFD100000 or not 0 < w1 >> 24 < 16:
                    continue
                after = [c for c in listed[k + 1 : k + 10] if c[0] >> 24 in {0xF0, 0xF3, 0xFD}]
                sized = [c for c in listed[k + 1 : k + 10] if c[0] >> 24 == 0xF2]
                if not after or after[0][0] >> 24 != 0xF3 or not sized:
                    continue  # a palette, or nothing that says how large
                size = ((sized[0][1] >> 12 & 0xFFF) >> 2) + 1, ((sized[0][1] & 0xFFF) >> 2) + 1
                if (w1 & 0xFFFFFF, *size) not in drawn.setdefault(w1 >> 24, []):
                    drawn[w1 >> 24].append((w1 & 0xFFFFFF, *size))
            named = sorted(self.by_start[v] for v in built if v and v in self.by_start)
            for pictures in drawn.values():
                if len(pictures) < 2:
                    continue
                for n in named:
                    data = self.file(n)
                    if any(off + 2 * w * h > len(data) for off, w, h in pictures):
                        continue
                    rows = [decode_texture(data, off, 0, 2, w, h) for off, w, h in pictures]
                    if all(_marker(r) for r in rows):
                        found += [r for r in rows if _capped(r)]
        if len(found) != 1:
            raise BuildError(
                f"the head that marks Link on a dungeon's map was not found in this ROM ({len(found)} fit)",
            )
        return [list(row) for row in found[0]]


def _marker(rows: Rows) -> bool:
    """Whether a picture is a small figure on a clear ground, as a map's marker is."""
    # Its corners clear, a quarter of it or more solid, and in every row and column the solid pixels in one run or
    # two. (On the first release the two markers have 153 and 94 of 256 pixels solid and two runs at most; the same
    # bytes of the other files the pause screen's code names give five runs or more in some row, or no clear corner,
    # or 11 solid pixels.)
    h, w = len(rows), len(rows[0])
    if any(rows[y][x][3] for y in (0, h - 1) for x in (0, w - 1)):
        return False
    if 4 * sum(1 for row in rows for px in row if px[3]) < w * h:
        return False
    for line in [*rows, *(list(column) for column in zip(*rows, strict=True))]:
        if sum(1 for k, px in enumerate(line) if px[3] and not (k and line[k - 1][3])) > 2:
            return False
    return True


def _capped(rows: Rows) -> bool:
    """Whether a marker is Link's head: a green cap, yellow hair and face, a dark outline, and the cap at the top."""
    # (On the first release: 27 pixels where green leads, 52 of yellow and skin, 40 dark; the skull beside it has no
    # green or yellow at all, and 14 dark.)
    green, yellow, dark, first = 0, 0, 0, None
    for row in rows:
        here = [0, 0]
        for r, g, b, a in row:
            if not a:
                continue
            if max(r, g, b) < 40:
                dark += 1
            elif g > 1.3 * r and g > 1.3 * b:
                here[0] += 1
            elif r > 150 and g > 100 and b < 0.6 * r:
                here[1] += 1
        green, yellow = green + here[0], yellow + here[1]
        if first is None and any(here):
            first = here
    return green >= 12 and yellow >= 25 and dark >= 20 and first[0] > first[1]
