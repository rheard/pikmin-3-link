"""
Read Link's voice out of an Ocarina of Time ROM.

Nothing from the ROM is kept anywhere: your own copy is read each time.

The game keeps its sound in three files, stored as they are: the fonts (which sample an instrument
or a sound effect plays, and how it is tuned), the sequences (scripts) and the samples (the
console's ADPCM, nine bytes for sixteen samples). The program has a table for each, and a table's
entries tile its file exactly, which is how the three are found.

A sound effect is asked for by a 16-bit number: the top digit is one of seven banks (6 is Link's
voice) and the low byte is the sound. Sequence 0 is the script that plays them all. It starts
sixteen channels, shared out among the banks by a table in the program. A channel waits to be given
a number, looks it up in a table of routines and runs the routine, which starts a layer that plays
one note. `Voice(rom)` follows exactly that path, running the script as the game does. It has to be
run and not just read: the routines write into the script as they go. The routine for a shout
writes how many recordings there are into a "pick a number" command, picks, and writes the
recording and its loudness into the note the layer then plays.

So most of what he says is several recordings, one picked each time. `takes(n)` runs the routine
once for every way its picks can fall and gives each outcome with its share of the times; `play(n)`
gives each recording as the game plays it: whole (a note of no length lasts as long as its sample),
at the rate its tuning gives, at the loudness of its note and the channel's envelope.

The numbers his code asks his voice for as he throws, as he swings his sword and as he spins are
read too, from the code that moves him: `thrown`, `swung` and `spun`.

What is left out: what the game does to a sound by where it is (panning, echo, the muffling of a
sound behind the camera, a little change of pitch its code adds to some); notes that are let go
before the sample ends, slides, vibrato and sounds that loop, all of which `play` refuses; the
fonts' instruments and drums (his voice is all sound effects); and the smaller ADPCM form (five
bytes for sixteen samples) that the second font's sound effects are kept in.

The formats and the commands' meanings follow the public documentation of the game's source. How
many bytes each channel command takes is read from the program's own table of them.
"""

from __future__ import annotations

import re
import struct

from dataclasses import astuple, dataclass, replace
from fractions import Fraction
from typing import TYPE_CHECKING, NamedTuple

from link.errors import BuildError
from link.oot import mips

if TYPE_CHECKING:
    from link.oot.rom import Rom

OUTPUT_RATE = 32000  # the game mixes at 32 kHz; a sample with tuning 1.0 plays at this rate
BANKS = 7  # kinds of sound effect: a sound number's top digit
EFFECTS, DRUMS = 0x7E, 0x7F  # the two instrument numbers that are not instruments
ADPCM = 0  # the one kind of sample decoded here

# The last fourteen entries of the program's table of how many bytes each command takes: the jumps,
# loops and waits (0xF2 to 0xFF). The table is found by them and read for all the rest.
_FLOW = bytes([0x01, 0x01, 0x01, 0x81, 0x00, 0x00, 0x01, 0x81, 0x81, 0x81, 0x81, 0x00, 0x00, 0x00])
_FIRST = 0xB0  # the first command the table has
# Bytes of operand for the commands of a sequence's opening, which the table does not cover.
_SEQUENCE = {
    0xF1: 1, 0xF0: 0, 0xDF: 1, 0xDE: 1, 0xDD: 1, 0xDC: 1, 0xDB: 1, 0xDA: 3, 0xD9: 1, 0xD7: 2, 0xD6: 2,
    0xD5: 1, 0xD4: 0, 0xD3: 1, 0xD2: 2, 0xD1: 2, 0xD0: 1, 0xCE: 1, 0xCD: 2, 0xCC: 1, 0xC9: 1, 0xC8: 1,
    0xC7: 3, 0xC6: 0, 0xC5: 2, 0xC4: 2,
}  # fmt: skip
# Channel commands that change nothing followed here: where a sound is heard from, its echo, the
# muffling filter, which notes may be taken from other channels.
_PLACING = {0xD0, 0xE9, 0xE5, 0xD4, 0xF1, 0xF0, 0xCA, 0xDD, 0xDC, 0xB0, 0xB1, 0xB3, 0xD1, 0xD2, 0xED, 0xBB}
_VIBRATO = {0xD8, 0xD7, 0xE1, 0xE2, 0xE3}


def _signed(v: int) -> int:
    """A byte as a signed number."""
    return ((v & 0xFF) ^ 0x80) - 0x80


def _var(d: bytes | bytearray, p: int) -> tuple[int, int]:
    """A length in a script, and where the script goes on: one byte, or two when the first has its top bit set."""
    v = d[p]
    if v & 0x80:
        return (v & 0x7F) << 8 | d[p + 1], p + 2
    return v, p + 1


class Note(NamedTuple):
    """One note a layer plays, with what the layer has set by then."""

    pitch: int  # the note's own number, 0 to 63
    ticks: int  # how long it lasts; 0 is as long as its sample
    velocity: int  # how loud
    gate: int  # how early it is let go
    transpose: int  # the layer's own transposition: for a sound effect, a step is 64 effects on
    instrument: int | None  # the layer's own instrument, if it chose one
    slide: tuple[int, int, int] | None  # a slide in pitch: (how, towards which note, how fast)
    legato: bool  # run together with the notes around it


@dataclass
class Settings:
    """What a channel has set for the layers it starts."""

    font: int
    large: bool = False  # its layers write their notes in the long form
    instrument: int | None = None
    transpose: int = 0
    envelope: int | None = None  # where its envelope is in the script
    release: int | None = None
    bend: int = 0
    scale: float = 1.0  # what its notes' rates are multiplied by
    volume: float = 1.0  # 0 to 1
    vibrato: bool = False


@dataclass
class Take:
    """
    One way a sound number can come out: `share` of the times, what each of its layers plays
    (`layers`, {layer: events}), the channel's settings as the layers start (`state`) and the
    routine the number names (`routine`, an address in the script).
    """

    share: Fraction
    layers: dict[int, tuple[tuple[str, Note | int], ...]]
    state: Settings
    routine: int | None


@dataclass(frozen=True)
class Sample:
    """A recording, as a font names it: where it is in the samples' file, how it is kept and how it is tuned."""

    address: int
    size: int  # in bytes
    kind: int  # how it is kept: `ADPCM` is the one kind decoded here
    bank: int  # the bank of the samples' file it is in
    tuning: float  # its rate, as a part of `OUTPUT_RATE`
    length: int  # in samples
    loops: bool
    book: tuple[int, int, tuple[int, ...]]  # the ADPCM codebook: order, predictors, their numbers


@dataclass
class Sound:
    """
    One recording a sound number may play, as the game plays it: `samples` (signed 16-bit, mono) at
    `rate` Hz, `share` of the times.

    `font` and `effect` say which sound effect of which font it is, and `sample` where its
    recording is. `velocity` is its note's loudness, of 127, and `gain` how loud the game plays the
    recording against how it is stored: the note's loudness, the channel's envelope and its volume.
    """

    samples: list[int]
    rate: int
    share: Fraction
    number: int
    font: int
    effect: int
    velocity: int
    gain: float
    sample: Sample


class _Picks:
    """
    The picks a routine is given in one run of it: those `fixed`, in order, and after them the
    first of each. Every other way a pick past those could fall is kept in `pending`, for a run of
    its own.
    """

    def __init__(self, fixed: list[int], pending: list[list[int]]) -> None:
        self.fixed, self.pending, self.taken = fixed, pending, []

    def __call__(self, n: int) -> int:
        """Pick a number below `n`."""
        if len(self.taken) < len(self.fixed):
            pick = self.fixed[len(self.taken)]
        else:
            pick = 0
            self.pending.extend([c for c, _ in self.taken] + [other] for other in range(1, n))
        self.taken.append((pick, n))
        return pick


class _Channel:
    """One channel of the sound-effect sequence, run as the game runs it, on a copy of the script."""

    def __init__(self, voice: Voice, number: int, choose: _Picks) -> None:
        self.voice, self.number, self.choose = voice, number, choose
        self.d = bytearray(voice.script)
        self.pc = voice.starts[number]
        self.stack = []  # where a call returns to; a loop's start and its count
        self.value = 0  # the script's working value, a signed byte
        self.pointer = 0  # ... and its one 16-bit value
        self.io = [-1] * 8  # what the game and the script hand each other
        self.other = {}  # what it hands other channels: {(channel, port): value}
        self.table = None
        self.layers = {}  # {layer: where its script is}
        self.routines = []  # the routines it has called by the table, in order
        self.state = Settings(font=voice.default_font)

    def _u16(self, at: int) -> int:
        return struct.unpack_from(">H", self.d, at)[0]

    def run(self, limit: int = 4000) -> int | None:
        """
        Run until the script waits.

        Returns:
            The ticks it waits for, or None if the channel ends.

        Raises:
            BuildError: If it meets a command not handled here, or never waits.
        """
        d, state = self.d, self.state
        for _ in range(limit):
            at = self.pc
            cmd = d[at]
            self.pc += 1
            if cmd < _FIRST:
                wait = self._low(cmd, at)
                if wait is not None:
                    return wait
                continue
            spec, args = self.voice.args[cmd - _FIRST], []
            for i in range(spec & 3):
                if spec & (0x80 >> i):
                    args.append(self._u16(self.pc))
                    self.pc += 2
                else:
                    args.append(d[self.pc])
                    self.pc += 1
            if cmd == 0xFF:
                if not self.stack:
                    return None
                self.pc = self.stack.pop()[0]
            elif cmd == 0xFE:
                return 1
            elif cmd == 0xFD:
                ticks, self.pc = _var(d, self.pc)
                return ticks
            elif cmd == 0xFC:
                self.stack.append([self.pc, None])
                self.pc = args[0]
            elif (
                cmd == 0xFB
                or (cmd == 0xFA and self.value == 0)
                or (cmd == 0xF9 and self.value < 0)
                or (cmd == 0xF5 and self.value >= 0)
            ):
                self.pc = args[0]
            elif cmd in {0xFA, 0xF9, 0xF5}:
                pass
            elif cmd == 0xF4 or (cmd == 0xF3 and self.value == 0) or (cmd == 0xF2 and self.value < 0):
                self.pc += _signed(args[0])
            elif cmd in {0xF3, 0xF2}:
                pass
            elif cmd == 0xF8:
                self.stack.append([self.pc, args[0]])
            elif cmd == 0xF7:
                self.stack[-1][1] = (self.stack[-1][1] - 1) & 0xFF
                if self.stack[-1][1]:
                    self.pc = self.stack[-1][0]
                else:
                    self.stack.pop()
            elif cmd == 0xF6:
                self.stack.pop()
            # -- the working value and the script itself
            elif cmd == 0xCC:
                self.value = _signed(args[0])
            elif cmd == 0xC8:
                self.value = _signed(self.value - args[0])
            elif cmd == 0xC9:
                self.value = _signed(self.value & args[0])
            elif cmd == 0xCB:
                self.value = _signed(d[args[0] + self.value])
            elif cmd == 0xC7:
                d[args[1]] = (self.value + args[0]) & 0xFF
            elif cmd == 0xCE:
                self.pointer = args[0]
            elif cmd == 0xCF:
                d[args[0]], d[args[0] + 1] = self.pointer >> 8, self.pointer & 0xFF
            elif cmd == 0xB8:
                if not args[0]:
                    raise BuildError("a routine of the sound-effect script picks from a range that is not given")
                self.value = self.choose(args[0])
            elif cmd == 0xC2:
                self.table = args[0]
            elif cmd == 0xC5:
                if self.value != -1:
                    self.table = self._u16(self.table + 2 * self.value)
            elif cmd == 0xE4:
                if self.value != -1:
                    self.stack.append([self.pc, None])
                    self.pc = self._u16(self.table + 2 * self.value)
                    self.routines.append(self.pc)
            # -- what the layers will play with
            elif cmd == 0xC1:
                state.instrument = args[0]
            elif cmd == 0xEB:
                state.font, state.instrument = self.voice.font_of(0, args[0]), args[1]
            elif cmd == 0xC6:
                state.font = self.voice.font_of(0, args[0])
            elif cmd in {0xC3, 0xC4}:
                state.large = cmd == 0xC4
            elif cmd == 0xDB:
                state.transpose = _signed(args[0])
            elif cmd == 0xDA:
                state.envelope = args[0]
            elif cmd == 0xD9:
                state.release = args[0]
            elif cmd in {0xD3, 0xEE}:
                state.bend = _signed(args[0])
            elif cmd == 0xDE:
                state.scale = args[0] / 32768.0
            elif cmd == 0xDF:
                state.volume = args[0] / 127.0
            elif cmd in _VIBRATO:
                state.vibrato = state.vibrato or any(args)
            elif cmd == 0xEC:
                state.vibrato = False
            elif cmd not in _PLACING:
                raise BuildError(f"channel command {cmd:#04x} at {at:#x} of the sound-effect script is not handled")
        raise BuildError("a channel routine of the sound-effect script does not come to a wait")

    def _low(self, cmd: int, at: int) -> int | None:
        """
        Run one of the commands below the table's: a port, another channel or a layer is in the low bits.

        Returns:
            The ticks the channel waits for, if the command is a wait; otherwise None.

        Raises:
            BuildError: If the command is not one handled here.
        """
        d = self.d
        kind, low = (cmd & 0xF8, cmd & 7) if cmd >= 0x70 else (cmd & 0xF0, cmd & 0xF)
        if kind == 0x00:
            return low
        if kind == 0x60 and low < 8:
            self.value = self.io[low]
            if low < 2:
                self.io[low] = -1  # (the game's two "a new sound, please" ports are read once)
        elif kind == 0x50 and low < 8:
            self.value = _signed(self.value - self.io[low])
        elif kind == 0x70:
            self.io[low] = self.value
        elif kind in {0x30, 0x40}:
            port = d[self.pc]
            self.pc += 1
            if kind == 0x40:
                self.value = self.io[port] if low == self.number else self.other.get((low, port), -1)
            elif low == self.number:
                self.io[port] = self.value
            else:
                self.other[low, port] = self.value
        elif kind == 0x80:
            self.value = 0 if low in self.layers else -1
        elif kind == 0x88:
            self.layers[low] = self._u16(self.pc)
            self.pc += 2
        elif kind == 0x78:
            self.pc += 2
            self.layers[low] = self.pc + struct.unpack_from(">h", d, self.pc - 2)[0]
        elif kind == 0x90:
            self.layers.pop(low, None)
        elif kind == 0x98:
            if self.value != -1:
                self.layers[low] = self._u16(self.table + 2 * self.value)
        else:
            raise BuildError(f"channel command {cmd:#04x} at {at:#x} of the sound-effect script is not handled")
        return None


def _events(d: bytes | bytearray, at: int, *, large: bool) -> tuple[tuple[str, Note | int], ...]:
    """
    What the layer whose script is at `at` plays, to its end.

    Returns:
        ("note", Note) and ("rest", ticks), in order.

    Raises:
        BuildError: If the layer uses a command not handled here, or never ends.
    """
    p, out, stack = at, [], []
    transpose, instrument, slide, legato, ticks = 0, None, None, False, 0
    for _ in range(400):
        here = p
        cmd = d[p]
        p += 1
        if cmd == 0xFF:
            if not stack:
                return tuple(out)
            p = stack.pop()[0]
        elif cmd == 0xC0:
            rest, p = _var(d, p)
            out.append(("rest", rest))
        elif cmd < 0xC0:
            if not large:
                raise BuildError(f"the layer at {at:#x} uses short notes, which are not handled")
            kind, gate = cmd >> 6, 0
            if kind != 2:  # kinds 0 and 1 carry their length; 2 reuses the last
                ticks, p = _var(d, p)
            velocity = d[p]
            p += 1
            if kind != 1:  # kinds 0 and 2 also say how early to let go
                gate = d[p]
                p += 1
            out.append(("note", Note(cmd & 0x3F, ticks, velocity, gate, transpose, instrument, slide, legato)))
        elif cmd == 0xC2:
            transpose, p = _signed(d[p]), p + 1
        elif cmd == 0xC6:
            instrument, p = d[p], p + 1
        elif cmd == 0xC7:  # a slide in pitch: how, towards which note, how fast
            mode, target = d[p], d[p + 1]
            p += 2
            if mode & 0x80:
                time, p = d[p], p + 1
            else:
                time, p = _var(d, p)
            slide = (mode, target, time)
        elif cmd == 0xC8:
            slide = None
        elif cmd in {0xC4, 0xC5}:
            legato = cmd == 0xC4
        elif cmd == 0xC3:
            p = _var(d, p)[1]
        elif cmd in {0xC1, 0xC9, 0xCA, 0xCD, 0xCE, 0xCF}:
            p += 1
        elif cmd == 0xCB:
            p += 3
        elif cmd == 0xCC or 0xD0 <= cmd <= 0xEF:
            pass
        elif cmd == 0xFC:
            stack.append([p + 2, None])
            p = struct.unpack_from(">H", d, p)[0]
        elif cmd == 0xFB:
            p = struct.unpack_from(">H", d, p)[0]
        elif cmd == 0xF4:
            p += 1 + _signed(d[p])
        elif cmd == 0xF8:
            stack.append([p + 1, d[p]])
            p += 1
        elif cmd == 0xF7:
            stack[-1][1] = (stack[-1][1] - 1) & 0xFF
            if stack[-1][1]:
                p = stack[-1][0]
            else:
                stack.pop()
        else:
            raise BuildError(f"layer command {cmd:#04x} at {here:#x} of the sound-effect script is not handled")
    raise BuildError(f"the layer at {at:#x} of the sound-effect script does not end")


class Voice:
    """Link's voice: the sound effects of his ROM, as its sound-effect script plays them."""

    def __init__(self, rom: Rom) -> None:
        self.rom = rom
        self._find_files()
        self._find_commands()
        self._find_font_lists()
        self._open()
        self._find_sharing()

    # -- finding things ------------------------------------------------------------------------------

    def _find_files(self) -> None:
        """
        Find the three sound files, and the program's table of each.

        A table is a count, fourteen bytes of nothing, then sixteen bytes an entry (where it starts
        in its file, how long it is, and three numbers). A table is known by its entries tiling one
        of the files stored as they are, from 0 to the file's end. The samples are the largest file;
        of the other two, the fonts' entries carry counts of instruments and the sequences' carry
        nothing.

        Raises:
            BuildError: If there are not three such tables, or the fonts' and the sequences' cannot
                be told apart.
        """
        rom, prog = self.rom, self.rom.program
        plain = {}  # {length: the files stored as they are that long}
        for i, (start, end, _, at_end) in enumerate(rom.files):
            if not at_end:
                plain.setdefault(end - start, []).append(i)
        found = []
        for at in range(0, len(prog) - 32, 4):
            if prog[at] or not prog[at + 1] or prog[at + 16 : at + 20] != bytes(4) or any(prog[at + 2 : at + 16]):
                continue
            count = prog[at + 1]
            if at + 16 + 16 * count > len(prog):
                continue
            entries = [struct.unpack_from(">IIbbHHH", prog, at + 16 + 16 * k) for k in range(count)]
            end = 0
            for off, size in sorted(e[:2] for e in entries if e[1]):
                if off != end:
                    break
                end = (off + size + 15) & ~15
            else:
                if len(plain.get(end, ())) == 1:
                    found.append((plain[end][0], at, entries))
        if len(found) != 3 or len({f for f, _, _ in found}) != 3:
            raise BuildError(f"the program's tables of fonts, sequences and samples were not found ({len(found)} fit)")
        found.sort(key=lambda t: len(rom.file(t[0])))
        (a, b), samples = found[:2], found[2]
        fonts, sequences = (a, b) if any(e[5] for e in a[2]) else (b, a)
        if not all(e[5] for e in fonts[2] if e[1]) or any(e[4] or e[5] or e[6] for e in sequences[2]):
            raise BuildError("the program's tables of fonts and sequences could not be told apart")
        self.font_file, _, self.fonts = fonts
        self.sequence_file, _, self.sequences = sequences
        self.sample_file, _, self.sample_banks = samples

    def _find_commands(self) -> None:
        """
        Find the program's table of the commands' operands, a byte for each command from 0xB0: how
        many it takes in the low two bits, and a bit from the top for each that is two bytes.

        Raises:
            BuildError: If it is not found once.
        """
        prog = self.rom.program
        hits = [m.start() for m in re.finditer(re.escape(_FLOW), prog)]
        if len(hits) != 1:
            raise BuildError(f"the program's table of script commands was not found ({len(hits)} fit)")
        at = hits[0] - (0x100 - len(_FLOW) - _FIRST)
        self.args = prog[at : at + 0x100 - _FIRST]

    def _find_font_lists(self) -> None:
        """
        Find which fonts each sequence may use: a 16-bit offset for each sequence, then the lists,
        each a count and that many fonts.

        Raises:
            BuildError: If they are not found once.
        """
        prog, n = self.rom.program, len(self.sequences)
        key, hits, at = struct.pack(">H", 2 * n), [], -1
        while True:
            at = prog.find(key, at + 1)
            if at < 0:
                break
            if at % 2 or at + 2 * n > len(prog):
                continue
            offsets = struct.unpack_from(f">{n}H", prog, at)
            if all(
                o >= 2 * n
                and at + o + 5 <= len(prog)
                and 1 <= prog[at + o] <= 4
                and all(f < len(self.fonts) for f in prog[at + o + 1 : at + o + 1 + prog[at + o]])
                for o in offsets
            ):
                hits.append(at)
        if len(hits) != 1:
            raise BuildError(f"the program's lists of each sequence's fonts were not found ({len(hits)} fit)")
        self.font_lists = hits[0]

    def font_of(self, sequence: int, index: int | None = None) -> int:
        """
        The font a channel's "use font `index`" picks: the lists are stored last first. With no
        index, the font a channel starts with: the last one loaded, which is the list's last.

        Returns:
            The font's number.

        Raises:
            BuildError: If the sequence has no such font.
        """
        prog = self.rom.program
        at = self.font_lists + struct.unpack_from(">H", prog, self.font_lists + 2 * sequence)[0]
        count = prog[at]
        if index is None:
            return prog[at + count]
        if index >= count:
            raise BuildError(f"sequence {sequence} has {count} fonts, not {index + 1}")
        return prog[at + count - index]

    def _open(self) -> None:
        """
        Run sequence 0's opening, for where each channel's script starts.

        Raises:
            BuildError: If it uses a command not handled here, or does not start its channels.
        """
        off, size = self.sequences[0][:2]
        self.script = d = bytes(self.rom.file(self.sequence_file)[off : off + size])
        self.default_font = self.font_of(0)
        self.starts = {}
        p = 0
        for _ in range(200):
            cmd = d[p]
            if cmd in {0xFF, 0xFD, 0xFE}:
                break
            if cmd >= 0xF2:
                spec = self.args[cmd - _FIRST]
                p += 1 + (spec & 3) * (2 if spec & 0x80 else 1)
            elif cmd >= 0xC0:
                if cmd not in _SEQUENCE:
                    raise BuildError(f"sequence command {cmd:#04x} at {p:#x} is not handled")
                p += 1 + _SEQUENCE[cmd]
            elif cmd & 0xF0 == 0x90:
                self.starts[cmd & 0xF] = struct.unpack_from(">H", d, p + 1)[0]
                p += 3
            else:
                raise BuildError(f"sequence command {cmd:#04x} at {p:#x} is not handled")
        if not self.starts or sorted(self.starts) != list(range(len(self.starts))):
            raise BuildError("the sound-effect sequence does not start its channels at its opening")

    def _find_sharing(self) -> None:
        """
        Find how the channels are shared out among the banks.

        It is four rows of seven in the program, one for each way of sharing (the first is the
        usual), each row how many channels each bank has. It is known by its first row adding up to
        the channels the sequence starts, and by the four rows that follow it, which say how many of
        those channels each bank may use at once: never more.

        Raises:
            BuildError: If it is not found once.
        """
        prog, total, hits = self.rom.program, len(self.starts), []
        for m in re.finditer(rb"(?=[\x01-\x08]{%d})" % BANKS, prog):
            at = m.start()
            rows = [prog[at + BANKS * k : at + BANKS * k + BANKS] for k in range(8)]
            if (
                sum(rows[0]) == total
                and all(sum(r) <= total for r in rows[1:4])
                and len(rows[7]) == BANKS
                and all(
                    u <= a for have, use in zip(rows[:4], rows[4:], strict=True) for a, u in zip(have, use, strict=True)
                )
                and all(any(r) for r in rows[4:])
            ):
                hits.append(at)
        if len(hits) != 1:
            raise BuildError(f"the program's table of each bank's channels was not found ({len(hits)} fit)")
        row = prog[hits[0] : hits[0] + BANKS]
        self.bank_channels = [list(range(sum(row[:b]), sum(row[: b + 1]))) for b in range(BANKS)]

    # -- the script ----------------------------------------------------------------------------------

    def _ask(self, number: int, choose: _Picks) -> _Channel:
        """
        Give a channel of the number's bank the number, as the game does, and run it until it has
        started its layers and waits.

        Returns:
            The channel.

        Raises:
            BuildError: If the number has no bank, or the bank's channel ends.
        """
        bank = number >> 12
        if not 0 <= bank < BANKS or not self.bank_channels[bank]:
            raise BuildError(f"{number:#x} is not a sound number: there is no bank {bank}")
        channel = _Channel(self, self.bank_channels[bank][0], choose)
        if channel.run() is None:
            raise BuildError(f"the channel of bank {bank} ends before it is asked for anything")
        # What the game sets: "a new sound", which one (its low byte, and for the large banks the
        # bit above it), and no muffling (a sound in front of the camera).
        channel.io[0], channel.io[4], channel.io[5], channel.io[3] = 1, _signed(number & 0xFF), number >> 8 & 1, 0
        if channel.run() is None:
            raise BuildError(f"the channel of bank {bank} ends when it is asked for {number:#x}")
        return channel

    def takes(self, number: int) -> list[Take]:
        """
        Every way a sound number can come out, each with its share of the times: the routine is run
        once for each way its picks can fall, every pick taken as equally likely.

        Returns:
            The takes, in the order of what their layers play.

        Raises:
            BuildError: If the number leads out of the script, comes out more than 64 ways, or is
                played in a way not handled here.
        """
        found, order, pending = {}, [], [[]]
        while pending:
            picks = _Picks(pending.pop(), pending)
            try:
                channel = self._ask(number, picks)
                large = channel.state.large
                layers = tuple(
                    (layer, _events(channel.d, at, large=large)) for layer, at in sorted(channel.layers.items())
                )
            except (IndexError, struct.error) as error:
                raise BuildError(f"{number:#x} leads out of the script: the bank has no such sound") from error
            share = Fraction(1)
            for _, n in picks.taken:
                share /= n
            key = (layers, astuple(channel.state), tuple(channel.routines[:1]))
            if key not in found:
                routine = channel.routines[0] if channel.routines else None
                found[key] = Take(Fraction(0), dict(layers), replace(channel.state), routine)
                order.append(key)
            found[key].share += share
            if len(order) > 64:
                raise BuildError(f"{number:#x} comes out more than 64 ways")
        return sorted((found[k] for k in order), key=lambda t: sorted(t.layers.items()))

    # -- fonts and samples ---------------------------------------------------------------------------

    def font(self, number: int) -> tuple[bytes, int]:
        """A font's own bytes, and how many sound effects it has."""
        off, size, _, _, _, _, effects = self.fonts[number]
        return self.rom.file(self.font_file)[off : off + size], effects

    def effect(self, font: int, index: int) -> Sample | None:
        """A font's sound effect: its sample, or None if it has none there."""
        data, count = self.font(font)
        if index >= count:
            return None
        at, tuning = struct.unpack_from(">If", data, struct.unpack_from(">I", data, 4)[0] + 8 * index)
        return self.sample(font, at, tuning) if at else None

    def sample(self, font: int, at: int, tuning: float) -> Sample:
        """
        The sample whose header is at `at` of a font: where it is in the samples' file, its kind, its
        length, its codebook and its tuning.

        Returns:
            The sample.

        Raises:
            BuildError: If its codebook is not one, or it is not inside a bank of samples.
        """
        data, _ = self.font(font)
        head, where, loop, book = struct.unpack_from(">4I", data, at)
        kind, medium, size = head >> 28, head >> 26 & 3, head & 0xFFFFFF
        _, end, count = struct.unpack_from(">IIi", data, loop)
        order, predictors = struct.unpack_from(">ii", data, book)
        if not (0 < order <= 8 and 0 < predictors <= 16):
            raise BuildError(f"font {font} has a codebook that is not one (order {order}, {predictors} predictors)")
        names = self.fonts[font][4]
        bank = (names >> 8, names & 0xFF)[medium] if medium < 2 else 0xFF
        if bank >= len(self.sample_banks):
            raise BuildError(f"font {font} keeps a sample in a bank it does not name")
        while not self.sample_banks[bank][1]:  # an empty entry stands for the one it names
            bank = self.sample_banks[bank][0]
        base, length = self.sample_banks[bank][:2]
        if where + size > length:
            raise BuildError(f"a sample of font {font} runs past the end of bank {bank}")
        book_numbers = struct.unpack_from(f">{8 * order * predictors}h", data, book + 8)
        return Sample(base + where, size, kind, bank, tuning, end, count != 0, (order, predictors, book_numbers))

    def decoded(self, sample: Sample) -> list[int]:
        """
        A sample's 16-bit samples, as it is stored. A sample's last frame may run a sample or so past
        the length its header gives, or stop that much short of it: what is given is the frames it
        has, cut to that length.

        Returns:
            The samples.

        Raises:
            BuildError: If it is kept in a form not decoded here.
        """
        if sample.kind != ADPCM:
            raise BuildError(f"a sample is kept in a form (kind {sample.kind}) that is not decoded here")
        return decode(self.rom.file(self.sample_file), sample.address, sample.size, sample.book)[: sample.length]

    def level(self, envelope: int) -> float:
        """
        The loudness a channel's envelope holds a note at, 0 to 1.

        Returns:
            The loudness.

        Raises:
            BuildError: If the envelope moves.
        """
        steps = [struct.unpack_from(">hh", self.script, envelope + 4 * i) for i in range(2)]
        if not (0 < steps[0][0] <= 4 and steps[1][0] == -1):
            raise BuildError(f"the envelope at {envelope:#x} does more than hold a level, which is not handled")
        return (steps[0][1] / 32767.0) ** 2

    def play(self, number: int) -> list[Sound]:
        """
        What a sound number plays, as the game plays it.

        Returns:
            A sound for each recording it may pick, with its share of the times.

        Raises:
            BuildError: If a take is not one whole sound effect.
        """
        out = []
        for take in self.takes(number):
            label = f"{number:#x}"
            notes = [e[1] for _, events in sorted(take.layers.items()) for e in events if e[0] == "note"]
            state = take.state
            if len(notes) != 1:
                raise BuildError(f"{label} plays {len(notes)} notes, which is not handled")
            note = notes[0]
            instrument = state.instrument if note.instrument is None else note.instrument
            if instrument != EFFECTS:
                what = "a drum" if instrument == DRUMS else "nothing" if instrument is None else "an instrument"
                raise BuildError(f"{label} plays {what}, not a sound effect, which is not handled")
            if note.ticks or note.gate:
                raise BuildError(f"{label} lets its note go before the sample ends, which is not handled")
            if note.slide or state.bend or state.vibrato:
                raise BuildError(f"{label} bends its pitch, which is not handled")
            if state.envelope is None:
                raise BuildError(f"{label} plays with an envelope that is not in the script, which is not handled")
            index = (note.transpose << 6) + note.pitch
            sample = self.effect(state.font, index)
            if sample is None:
                raise BuildError(f"{label} names sound effect {index} of font {state.font}, which is not there")
            if sample.loops:
                raise BuildError(f"{label} loops, which is not handled")
            gain = (min(note.velocity, 127) / 127.0) ** 2 * self.level(state.envelope) * state.volume
            samples = [max(-32768, min(32767, round(v * gain))) for v in self.decoded(sample)]
            rate = round(OUTPUT_RATE * sample.tuning * state.scale)
            out.append(Sound(samples, rate, take.share, number, state.font, index, note.velocity, gain, sample))
        return out

    # -- his code ------------------------------------------------------------------------------------

    def thrown(self, file: int) -> tuple[int, int]:
        """
        The two sound numbers his code asks for as he throws what he is holding.

        `file` is the file of the code that moves him (`Link.move_table` finds it). The routines are
        found by what they do. Each gives a speed forward and a speed upward to something whose
        address Link keeps (two numbers stored 0x68 and 0x60 into it, where every object of his game
        keeps those), and straight after has Link make two sounds: the first by a routine of the
        program's, the second by one of this file's, which is his voice. To the voice's number that
        routine adds one kept for each age; the adult's, the first, is added here. Every such routine
        must ask for the same two.

        Returns:
            (the swish, his voice).
        """
        swish, voice, _, age = self._throw(file)
        return swish, (voice + age) & 0xFFFF

    def _throw(self, file: int) -> tuple[int, int, int, int]:
        """
        Read the routines in which he throws, and his voice's routine.

        Returns:
            (the swish, his voice as the routines ask for it, the call to his voice's routine as the
            code makes it, the number that routine adds for the adult).

        Raises:
            BuildError: If the routines are not found or do not agree, or the voice's routine does
                not add an age's number the way expected.
        """
        rom = self.rom
        code = rom.file(file)
        words = mips.words_of(code)
        asked = set()
        for a, b in mips.routines(words):
            stored = {}  # register -> {where a float was stored through it: the instruction}
            for i in range(a, b):
                if words[i] >> 26 == 0x39:  # swc1
                    stored.setdefault(words[i] >> 21 & 31, {})[words[i] & 0xFFFF] = i
            for held, places in stored.items():
                if 0x68 not in places or 0x60 not in places:
                    continue
                after = max(places[0x68], places[0x60])
                # a sound: "the maker into the first argument", a call, "the number into the second"
                sounds = [
                    (words[i - 1] >> 21 & 31, words[i], words[i + 1] & 0xFFFF)
                    for i in range(after + 1, b - 1)
                    if words[i] >> 26 == 3
                    and words[i + 1] >> 16 == 0x2405
                    and words[i - 1] >> 26 == 0
                    and words[i - 1] & 0x7FF in {0x21, 0x25}
                    and words[i - 1] >> 11 & 31 == 4
                    and not words[i - 1] >> 16 & 31
                ]
                if len(sounds) != 2 or sounds[0][0] != sounds[1][0] or sounds[0][0] == held:
                    continue
                maker = sounds[0][0]
                if not any(
                    words[i] >> 26 == 0x23 and words[i] >> 16 & 31 == held and words[i] >> 21 & 31 == maker
                    for i in range(a, after)
                ):
                    continue  # the thing is not one whose address the maker keeps
                asked.add((sounds[0][2], sounds[1][2], sounds[0][1], sounds[1][1]))
        if len(asked) != 1:
            raise BuildError(f"the routines in which Link throws were not found in file {file}, or do not agree")
        swish, voice, _, call = asked.pop()
        # Where the file sits in memory, from the program's own entry for it: its place in the ROM
        # and then the addresses it is loaded between.
        at = rom.program.find(struct.pack(">2I", *rom.files[file][:2]))
        low, high = struct.unpack_from(">2I", rom.program, at + 8) if at >= 0 else (0, 0)
        target = (call & 0x3FFFFFF) << 2 | 0x80000000
        if not (low <= target < high and high - low >= len(code)):
            raise BuildError(f"the routine of Link's voice is not in file {file}, or where the file goes was not found")
        # The voice's routine: "load a 16-bit number from the age's entry, add it to the number".
        i = (target - low) // 4
        end = min(s for s in [*mips.starts(words), len(words)] if s > i)
        adds = [
            (words[k] & 0xFFFF, words[k] >> 21 & 31)
            for k in range(i, end - 2)
            if words[k] >> 26 == 0x25
            and any(
                words[j] >> 26 == 0
                and words[j] & 0x7FF == 0x21
                and words[j] >> 11 & 31 == 5
                and words[k] >> 16 & 31 in {words[j] >> 21 & 31, words[j] >> 16 & 31}
                for j in (k + 1, k + 2)
            )
        ]
        if len(adds) != 1:
            raise BuildError("the routine of Link's voice does not add an age's number the way expected")
        within, entry = adds[0]
        slots = {words[k] & 0xFFFF for k in range(i, end) if words[k] >> 26 == 0x23 and words[k] >> 16 & 31 == entry}
        if len(slots) != 1:
            raise BuildError("where Link keeps his age's entry was not found in the routine of his voice")
        slot = slots.pop()
        # The table of the ages' entries: wherever that pointer is set, it is the table's address
        # plus a multiple of the age.
        tables = set()
        for k, w in enumerate(words):
            if w >> 26 != 0x2B or w & 0xFFFF != slot:
                continue
            window = range(max(0, k - 24), k)
            for j in window:
                if words[j] >> 26 == 0 and words[j] & 0x7FF == 0x21 and words[j] >> 11 & 31 == w >> 16 & 31:
                    for reg in (words[j] >> 21 & 31, words[j] >> 16 & 31):
                        upper = [
                            words[m] & 0xFFFF for m in window if words[m] >> 26 == 0x0F and words[m] >> 16 & 31 == reg
                        ]
                        lower = [
                            words[m] & 0xFFFF
                            for m in window
                            if words[m] >> 26 == 0x09 and words[m] >> 16 & 31 == reg and words[m] >> 21 & 31 == reg
                        ]
                        if len(upper) == 1 and len(lower) == 1:
                            tables.add((upper[0] << 16) + ((lower[0] ^ 0x8000) - 0x8000))
        if len(tables) != 1 or not low <= next(iter(tables)) < low + len(code) - within - 2:
            raise BuildError(f"the table of the ages' entries was not found in file {file}")
        ages = tables.pop() - low  # where the entries are in the file
        return swish, voice, call, struct.unpack_from(">H", code, ages + within)[0]

    def swung(self, file: int) -> tuple[int, int, list[int]]:
        """
        His shout as he swings his sword, and his shout with the later blow of a run of swings.

        The routine is found by what it does: it is the one routine of the code that moves him that
        puts either of two numbers of his voice into one register, and then calls his voice's
        routine with that register and not with a number written there. On the first release it
        puts the second in after reading a count and finding it three or more (which the program's
        source as publicly documented has as the swings made in a run). Beside the voice it chooses
        a number of another bank the same way, for the blade.

        Returns:
            (the swing's shout, the later blow's, [the swishes of the blade]): sound numbers, as
            `thrown` gives them.
        """
        (first, later), blades, _ = self._swing(file)
        age = self._throw(file)[3]
        return (first + age) & 0xFFFF, (later + age) & 0xFFFF, blades

    def _swing(self, file: int) -> tuple[list[int], list[int], tuple[int, int]]:
        """
        Read the routine that chooses the sounds of a swing.

        Returns:
            ([the two numbers of his voice, as the routine has them], [the blade's], (where the
            routine starts, where it ends), in words of the file).

        Raises:
            BuildError: If not one routine of the file chooses that way.
        """
        _, voice, call, age = self._throw(file)
        bank = ((voice + age) & 0xFFFF) >> 12
        words = mips.words_of(self.rom.file(file))
        found = []
        for a, b in mips.routines(words):
            if call not in words[a:b]:
                continue
            loads = {}  # register -> the numbers written into it as they are
            for i in range(a, b):
                if words[i] >> 26 == 9 and not words[i] >> 21 & 31:
                    loads.setdefault(words[i] >> 16 & 31, []).append(words[i] & 0xFFFF)
            # (register 5 is where his voice's routine takes its number: a routine that writes numbers
            # straight into it is making several calls, not choosing between two for one)
            voices = [
                sorted(set(v))
                for reg, v in loads.items()
                if reg != 5 and len(set(v)) == 2 and all(n >> 12 == bank for n in v)
            ]
            if len(voices) == 1 and len(loads) > 1:
                blades = [
                    sorted({n for n in v if n})
                    for v in loads.values()
                    if len({n >> 12 for n in v if n}) == 1 and all(n >> 12 not in {0, bank} for n in v if n) and any(v)
                ]
                found.append((voices[0], blades, (a, b)))
        if len(found) != 1 or len(found[0][1]) != 1:
            raise BuildError(f"the routine choosing a swing's sounds was not found in file {file} ({len(found)} fit)")
        voices, (blades,), where = found[0]
        return voices, blades, where

    def spun(self, file: int) -> tuple[int, int]:
        """
        His shout as he spins, read from the routine that chooses the sounds of a swing (`swung`).

        That routine compares the swing's record of the table of swings with a number, and from
        that record on asks for no swish of the blade and for one number of his voice. On the first
        release the record is 24 and the shout the later blow's (0x6801); records 24 to 27 are the
        ones `Link.spins` reads as the spins.

        Returns:
            (the first record of the table of swings that his game takes as a spin, the number of his
            voice it asks for then, as `thrown` gives them).

        Raises:
            BuildError: If that routine does not tell a spin from a swing that way.
        """
        _, voice, _, age = self._throw(file)
        bank = ((voice + age) & 0xFFFF) >> 12
        a, b = self._swing(file)[2]
        words = mips.words_of(self.rom.file(file))
        found = []
        for i in range(a, b - 2):
            w = words[i]
            if w >> 26 != 0x0A or w >> 16 & 31 != 1:  # slti at, r, N
                continue
            branch = words[i + 1]
            if branch >> 26 not in {0x14, 0x15, 0x04, 0x05} or {branch >> 21 & 31, branch >> 16 & 31} != {0, 1}:
                continue
            # the way on when the record is N or more: nothing for the blade, and a number of his voice
            after = words[i + 3 : i + 7]
            said = [v & 0xFFFF for v in after if v >> 26 == 9 and not v >> 21 & 31 and (v & 0xFFFF) >> 12 == bank]
            quiet = any(v & 0xFFFF07FF in {0x21, 0x25} for v in after)  # move r, zero (an or or an addu of nothing)
            if len(said) == 1 and quiet:
                found.append((w & 0xFFFF, said[0]))
        if len(found) != 1:
            raise BuildError(f"where a swing's sounds are told from a spin's was not found ({len(found)} fit)")
        return found[0][0], (found[0][1] + age) & 0xFFFF


def decode(d: bytes, addr: int, size: int, book: tuple[int, int, tuple[int, ...]]) -> list[int]:
    """
    ADPCM to 16-bit samples: each nine-byte frame is a header and sixteen four-bit residuals.

    Returns:
        The samples.

    Raises:
        BuildError: If a frame names a predictor the codebook does not have.
    """
    order, predictors, flat = book
    # Each predictor is `order` rows of eight; the decoder wants, for each of the eight outputs of a
    # half frame, the weights of the `order` previous samples and of the residuals before it.
    tables = []
    for i in range(predictors):
        rows = [[0] * (order + 8) for _ in range(8)]
        for j in range(order):
            for k in range(8):
                rows[k][j] = flat[(i * order + j) * 8 + k]
        for k in range(1, 8):
            rows[k][order] = rows[k - 1][order - 1]
        rows[0][order] = 2048
        for k in range(1, 8):
            for j in range(k, 8):
                rows[j][k + order] = rows[j - k][order]
        tables.append(rows)
    out = []
    state = [0] * 16
    for f in range(size // 9):
        p = addr + 9 * f
        head = d[p]
        scale, index = 1 << (head >> 4), head & 0xF
        if index >= predictors:
            raise BuildError(f"frame {f} of a sample names predictor {index} of {predictors}")
        rows = tables[index]
        resid = [(nib - 16 if nib >= 8 else nib) * scale for b in d[p + 1 : p + 9] for nib in (b >> 4, b & 0xF)]
        new = [0] * 16
        for half in range(2):
            vec = (state[16 - order :] if half == 0 else new[8 - order : 8]) + resid[half * 8 : half * 8 + 8]
            for k in range(8):
                row = rows[k]
                total = 0
                for j in range(order + 8):
                    total += row[j] * vec[j]
                new[half * 8 + k] = total >> 11  # floor, as the game's decoder does
        state = [max(-32768, min(32767, v)) for v in new]
        out += state
    return out
