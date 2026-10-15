"""
Read Link out of an Ocarina of Time ROM: his model, its textures, what he holds and his animations.

`Link(rom)` gives:

* `parts`: his 21 limbs in the order animations address them, each with its parent, its offset from
  the parent and the triangles that move with it. The model's own space is used throughout: Y up,
  facing +Z, his left at +X, which is also how the Pikmin 3 leaders are laid out.
* `pose(rotations, root)`: every part's (rotation, position) in model space for a set of angles.
  With nothing given it is a rest pose (standing, arms straight out, soles on the ground) for
  binding a skeleton; the game itself has no rest pose, only animations.
* `skin(pose)`, `image(material)`: what to draw and with what.
* `animation("walk")`: an animation by what he does in it, or by number. His animations have no
  names in the ROM. Which is which comes from the table kept by the code that moves him, a file of
  its own, and each one used is checked by what it does.

How he differs from a simpler model:

* **A limb's list draws more than the limb.** Each drawn limb has a matrix, and a list switches
  between them as it goes, so that the vertices round a joint can follow the limb on the other side
  of it (the head's list starts with six vertices that follow the torso). A triangle belongs to the
  limb most of its corners follow; each corner keeps its own.
* **Real textures, built by a formula.** His surfaces are small textures (a palette of 256 colours
  for most), repeated, mirrored or clamped as each list sets them, and multiplied by a colour: the
  list's own, or the tunic's. `image` gives the result of the formula for the stretch of texture a
  material's triangles use, already unrolled.
* **What he holds is chosen by the program.** The skeleton's own lists are open hands and the
  scabbard with the sword in it. The program keeps tables of other lists for the hands, the scabbard
  and the waist, and picks by what he has out (`KINDS`).
"""

from __future__ import annotations

import functools
import math
import struct

from typing import TYPE_CHECKING, TypeVar

from link.errors import BuildError
from link.geometry import IDENTITY, Matrix, Vector, apart, euler_matrix, matmul, rotate, transpose
from link.oot import mips
from link.oot.animation import PARTS, Animation, Keys, Sequence
from link.oot.graphics import (
    G_CULL_BACK,
    G_CULL_FRONT,
    G_LIGHTING,
    G_TEXTURE_GEN,
    Image,
    Material,
    Pixel,
    Tile,
    Vertex,
    bleed,
    combiner,
)
from link.oot.rom import KEEP

if TYPE_CHECKING:
    from collections.abc import Callable

    from link.oot.rom import Rom

Pose = list[tuple[Matrix, Vector]]  # the rotation and position of every part, in the order of `parts`
SkinCorner = tuple[Vector, Vector, tuple[float, float], str]  # position, normal, (u, v), the part it follows
SkinTriangle = tuple[Material, str, list[SkinCorner]]

EYES = ("open", "half", "closed", "left", "right", "shock", "down", "tight")
MOUTHS = ("closed", "half", "open", "smile")

# The kinds of list the program swaps in for a hand, the scabbard or the waist, in the order of its
# own table of them (the order is the program's, as its source is publicly documented; three of the
# 21 are checked against the skeleton when the table is found).
KINDS = (
    "left open", "left closed", "left sword", "left sword 2", "left big sword", "left hammer", "left boomerang",
    "left bottle", "right open", "right closed", "right shield", "right bow", "right bow 2", "right ocarina",
    "right ocarina of time", "right hookshot", "sheath sword", "sheath empty", "sheath sword shield",
    "sheath shield", "waist",
)  # fmt: skip
SHIELDS = ("none", "deku", "hylian", "mirror")
SHIELD = "hylian"
BY_SHIELD = ("right shield", "sheath sword shield", "sheath shield")  # the kinds with a list for each shield
LOOK_LIMBS = {"sheath": "sheath", "left": "left_hand", "right": "right_hand"}  # a kind's first word -> its limb

# A rest pose, in degrees about each limb's own X, Y, Z (applied in that order). Every limb is
# modelled lying along its own +X, so with no rotation at all the model is a tangle; these angles
# stand him up square, legs straight down, arms straight out to the sides, palms down, feet and face
# forward. They were read off his own standing animations and squared up; the scabbard's is the
# angle it hangs at across his back there.
REST = {
    "waist": (-90.0, 0.0, -90.0), "upper": (90.0, 0.0, 90.0),
    "head": (0.0, 0.0, 90.0), "hat": (0.0, 0.0, 90.0), "sheath": (-90.0, 150.0, 0.0),
    "left_upperarm": (0.0, -90.0, 180.0), "right_upperarm": (0.0, 90.0, 180.0),
    "left_hand": (0.0, 0.0, -90.0), "right_hand": (0.0, 0.0, -90.0),
    "left_foot": (0.0, 0.0, -90.0), "right_foot": (0.0, 0.0, -90.0),
}  # fmt: skip

# The table of his animations in the code that moves him: rows of six, a row for each thing he does
# and a column for what he has in his hands (the first is nothing, the second the sword and shield).
# Its rows' order is the program's, as its source is publicly documented; each one used is checked
# by what it does. After the rows come four jumps of three animations each, and then fourteen pairs
# of things he does when left standing, each for a sword in hand and then for empty hands.
MOVES = ("wait", "walk", "run")
HANDS = 6
JUMPS = 4
IDLE_PAIRS = 14
ROLL_ROW = 16  # his roll, as he lands a long jump
CARRY_ROW, THROW_ROW = 13, 29  # he raises a thing over his head; he throws what he holds there
PUT_ROW = 30  # he sets down what he holds over his head
IDLES = {"glance": 10, "stretch": 4, "boot": 11, "shield": 12, "flourish": 9}  # pairs of the left-standing ones
ARMED = "_armed"  # what a move's name ends in for its animation with the sword and shield out

# The swings of his sword are a table of their own in the same code: 16 bytes a swing, which are
# three animations (the swing; how he comes back from it to his stand; how he comes back with
# something targeted) and the first and last frame on which the blade can hit. By the program's
# source as publicly documented it begins with four kinds, each four times over: with one hand and
# with two, and then both again for the later blow of a run of them.
SWINGS = ("forward", "right", "left", "stab")
SWING_BYTES = 16
# After that table the code keeps what a charge of the sword is made of: six pairs of animations
# (with one hand, then two), then two pairs of bytes that name the records of the table that are
# the spin and the bigger spin. Which pair is which was read from the routines that use them.
CHARGES = ("charge_in", "charge_in_right", "charge", "charge_out", "charge_walk", "charge_side")

# Segments a list may name that are not files: the program points them somewhere as it draws.
SEG_OBJECT, SEG_KEEP, SEG_EYES, SEG_MOUTH, SEG_CULL, SEG_MATRIX = 0x06, 0x04, 0x08, 0x09, 0x0C, 0x0D
# Display-list commands that change nothing a triangle is drawn with.
QUIET = (0xE6, 0xE7, 0xE8, 0xE9, 0x00, 0xD8, 0xDB, 0xE1)

Found = TypeVar("Found")


def _found_once(read: Callable[[Link], Found]) -> Callable[[Link], Found]:
    """A reading of his made once for the ROM and kept on it (`Rom.found`): none depends on what he holds."""

    @functools.wraps(read)
    def once(self: Link) -> Found:
        if read.__name__ not in self.rom.found:
            self.rom.found[read.__name__] = read(self)
        return self.rom.found[read.__name__]

    return once


class Part:
    """A limb, or a thing hung on him: its parent, its offset from it, and the triangles read for it."""

    def __init__(self, name: str, parent: int | None, offset: tuple[int, int, int]) -> None:
        self.name, self.parent, self.offset = name, parent, offset
        self.tris = []  # (material, corners, both sides?), in the part's own space


class _State:
    """What a display list has set by the time it draws a triangle."""

    def __init__(self, env: Pixel) -> None:
        self.verts = {}
        self.geometry = G_CULL_BACK | G_LIGHTING
        self.timg = (0, 0, 0)  # address, format, size of the image named last
        self.tlut = 0
        self.loaded = {}  # where in texture memory -> the image loaded there
        self.tiles = [Tile() for _ in range(8)]
        self.tex_on, self.tex_tile, self.tex_scale = False, 0, (1.0, 1.0)
        self.combine = (0, 0)
        self.prim, self.env = (255, 255, 255, 255), env
        self.mode = 0
        self.two_cycle = True  # how the game has set things when it draws an actor


class Link:
    """
    Adult Link as his game draws him, with the Hylian Shield.

    `holding` is the kinds of list for his scabbard and his two hands (`KINDS`). `looks` names other
    things the scabbard or a hand can be drawn as, {name: kind}: each is read as well, beside the
    three chosen, and `skin(pose, looks=True)` hands every one of them over under its own name, for
    a model that shows one of them at a time. They share the materials of everything else.
    """

    def __init__(
        self,
        rom: Rom,
        eyes: str = "open",
        mouth: str = "closed",
        holding: tuple[str, str, str] = ("sheath sword shield", "left open", "right open"),
        looks: dict[str, str] | None = None,
    ) -> None:
        self.rom = rom
        self.number = rom.link_objects[0]
        self.object = rom.object(self.number)
        self.keep = rom.object(KEEP)
        self._find_skeleton()
        self._find_tables()
        self._find_kinds()
        self._find_animations()
        self.eyes, self.mouth = EYES.index(eyes), MOUTHS.index(mouth)
        self.env = (*self.green, 255)  # the tunic's colour: the Kokiri's green
        self.materials = {}
        self.unknown = {}  # display-list commands met that this reader does not know: {number: times}
        self.choose(*holding, looks=looks)

    # -- finding things ------------------------------------------------------------------------------

    def _find_skeleton(self) -> None:
        """Find the skeleton's header: where its table of limbs is, how many, and how many are drawn."""
        d, found = self.object, []
        for at in range(0, len(d) - 8, 4):  # the header is nine bytes, and may be the file's last thing
            table, count, drawn = struct.unpack_from(">IB3xB", d, at)
            if table >> 24 != SEG_OBJECT or count != len(PARTS) or not 0 < drawn <= count:
                continue
            table &= 0xFFFFFF
            if table + 4 * count > len(d) or any(d[table + 4 * i] != SEG_OBJECT for i in range(count)):
                continue
            found.append((at, table, drawn))
        if len(found) != 1:
            raise BuildError(f"expected one skeleton of {len(PARTS)} limbs in Link's object, found {len(found)}")
        _, table, self.drawn = found[0]
        self.limbs = []
        for i in range(len(PARTS)):
            at = struct.unpack_from(">I", d, table + 4 * i)[0] & 0xFFFFFF
            x, y, z, child, sibling, near, far = struct.unpack_from(">3hBBII", d, at)
            self.limbs.append({"offset": (x, y, z), "child": child, "sibling": sibling, "near": near, "far": far})
        # parents, and the order the game visits the limbs in: a limb, its children, then its next sibling
        parent, order = {}, []

        def visit(i: int, above: int | None) -> None:
            parent[i] = above
            order.append(i)
            limb = self.limbs[i]
            if limb["child"] != 0xFF:
                visit(limb["child"], i)
            if limb["sibling"] != 0xFF:
                visit(limb["sibling"], above)

        visit(0, None)
        if order != list(range(len(PARTS))):
            raise BuildError("Link's limbs are not stored in the order the game visits them")
        self.parent = [parent[i] for i in range(len(PARTS))]
        # the matrix each drawn limb is given, in that order: what a list means by "matrix 7"
        self.matrix_limb = [i for i in order if self.limbs[i]["near"]]
        if len(self.matrix_limb) != self.drawn:
            raise BuildError(
                f"Link's skeleton says {self.drawn} limbs are drawn, and {len(self.matrix_limb)} have a list",
            )

    def _pointer_runs(self, low: int, high: int, least: int) -> list[tuple[int, list[int]]]:
        """Runs in the program of `least` or more addresses in a row, each in his object between `low` and `high`."""
        d, runs, at = self.rom.program, [], 0
        while at < len(d) - 4:
            if d[at] == SEG_OBJECT:
                run, p = [], at
                while p < len(d) - 4:
                    v = struct.unpack_from(">I", d, p)[0]
                    if v >> 24 != SEG_OBJECT or not low <= (v & 0xFFFFFF) < high or v & 3:
                        break
                    run.append(v)
                    p += 4
                if len(run) >= least:
                    runs.append((at, run))
                at = p + 4 if run else at + 4
            else:
                at += 4
        return runs

    def _find_tables(self) -> None:
        """
        Find what the program keeps of his face, and his tunic's green.

        His eyes are eight pictures of 64x32, one byte a texel, one after another, and his mouths
        four of 32x32: the program keeps a table of each, straight after the other. The green is
        the colour the hat's own list sets before it draws, which is the Kokiri tunic's.

        Raises:
            BuildError: If either is not found.
        """
        self.eye_table = self.mouth_table = None
        for _, run in self._pointer_runs(0, len(self.object), 12):
            for k in range(len(run) - 11):
                eyes, mouths = run[k : k + 8], run[k + 8 : k + 12]
                if all(eyes[i + 1] - eyes[i] == 0x800 for i in range(7)) and all(
                    mouths[i + 1] - mouths[i] == 0x400 for i in range(3)
                ):
                    self.eye_table, self.mouth_table = eyes, mouths
                    break
            if self.eye_table:
                break
        if not self.eye_table:
            raise BuildError("the program's table of Link's eyes and mouths was not found")
        d = self.object
        p = self.limbs[PARTS.index("hat")]["near"] & 0xFFFFFF
        while d[p] != 0xDF:
            if d[p] == 0xFA:
                self.green = tuple(d[p + 4 : p + 7])
                return
            p += 8
        raise BuildError("the list of Link's hat sets no colour")

    def _find_kinds(self) -> None:
        """
        Find the program's table of the 21 kinds of list it swaps in (`KINDS`).

        It is 21 addresses in the program's own memory, one after another, each of a table of
        lists. Where the program sits in memory is not assumed: it is worked out from the one kind
        that must hold the skeleton's own waist list, and then the open hands, which must hold the
        skeleton's own hands, are the check.

        Raises:
            BuildError: If no such table is found.
        """
        d = self.rom.program
        own = {
            KINDS.index(kind): struct.pack(">I", self.limbs[PARTS.index(part)]["near"])
            for kind, part in (("waist", "waist"), ("left open", "left_hand"), ("right open", "right_hand"))
        }
        waist, places = own[KINDS.index("waist")], []
        p = d.find(waist)
        while p >= 0:
            if p % 4 == 0:
                places.append(p)
            p = d.find(waist, p + 4)
        for at in range(0, len(d) - 4 * len(KINDS), 4):
            if d[at] != 0x80:
                continue
            words = struct.unpack_from(f">{len(KINDS)}I", d, at)
            if any(w >> 24 != 0x80 or w & 3 for w in words) or max(words) - min(words) > 0x1000:
                continue
            for place in places:
                base = words[KINDS.index("waist")] - place
                tables = [w - base for w in words]
                if all(0 <= t <= len(d) - 16 for t in tables) and all(
                    d[tables[k] : tables[k] + 4] == v for k, v in own.items()
                ):
                    self.kinds, self.program_base = tables, base
                    return
        raise BuildError("the program's table of the lists for Link's hands, scabbard and waist was not found")

    def held(self, kind: str) -> int:
        """The list of one of the `KINDS`, for adult Link with the Hylian Shield."""
        row = SHIELDS.index(SHIELD) if kind in BY_SHIELD else 0
        return struct.unpack_from(">I", self.rom.program, self.kinds[KINDS.index(kind)] + 4 * (4 * row))[0]

    def choose(self, sheath: str, left: str, right: str, looks: dict[str, str] | None = None) -> None:
        """
        Read the model with these kinds of list for the scabbard and the hands, and these looks.

        Raises:
            BuildError: If a kind is not one for its limb, or this Link has no list of it.
        """
        for kind, side in ((sheath, "sheath"), (left, "left"), (right, "right")):
            if not kind.startswith(side):
                raise BuildError(f"{kind} is not a kind of list for the {side}")
        self.holding = (sheath, left, right)
        self.chosen = {
            "sheath": self.held(sheath),
            "left_hand": self.held(left),
            "right_hand": self.held(right),
            "waist": self.held("waist"),
        }
        if not all(self.chosen.values()):
            raise BuildError("Link has no list for one of: {}, {}, {}".format(*self.holding))
        self.looks = dict(looks or {})
        for name, kind in self.looks.items():
            if kind.split()[0] not in LOOK_LIMBS or name in PARTS or not self.held(kind):
                raise BuildError(f"{name}: Link has no list of the kind {kind} for his scabbard or a hand")
        self._build()

    # -- animations ----------------------------------------------------------------------------------

    def _find_animations(self) -> None:
        """
        Find his animations' headers: the longest run of them in the shared object, eight bytes
        each (a frame count, a zero, and where the frames are in segment 7). And the file the frames
        are in: the one, stored as it is, that the last frame ends at the end of.

        Raises:
            BuildError: If either is not found.
        """
        keep, best, at = self.keep, (0, 0), 0
        while at < len(keep) - 8:
            p = at
            while p < len(keep) - 8:
                n, zero, where = struct.unpack_from(">hhI", keep, p)
                if n <= 0 or zero or where >> 24 != 0x07:
                    break
                p += 8
            if p - at > best[1] - best[0]:
                best = (at, p)
            at = max(p, at + 4)
        self.animations = list(range(best[0], best[1], 8))
        if len(self.animations) < 100:
            raise BuildError("Link's animations were not found in the shared object")
        end = max(
            (struct.unpack_from(">I", keep, h + 4)[0] & 0xFFFFFF)
            + struct.unpack_from(">h", keep, h)[0] * Animation.FRAME
            for h in self.animations
        )
        files = [i for i, e in enumerate(self.rom.files) if not e[3] and 0 <= e[1] - e[0] - end < 16]
        if len(files) != 1:
            raise BuildError(f"the file of Link's animation frames was not found ({len(files)} fit)")
        self.animation_file = files[0]

    def animation_data(self) -> bytes:
        """The file his animations' frames are in."""
        return self.rom.file(self.animation_file)

    def animation(self, which: str | int) -> Animation:
        """An animation by what he does in it, by its place in the list of headers, or by a header's address."""
        if isinstance(which, str):
            found = self.moves()
            if which not in found:
                for prefix, table in (
                    ("roll", self.dodges),
                    (("lift", "hold", "throw", "put"), self.carries),
                    ("idle_", self.idles),
                    ("sit", lambda: self.sits()[0]),
                    ("bow_", self.bows),
                ):
                    if which.startswith(prefix):
                        return Animation(self, table()[which])
            return Animation(self, found[which])
        return Animation(self, self.animations[which] if which < len(self.animations) else which)

    @_found_once
    def move_table(self) -> tuple[int, int, list[int]]:
        """
        Find the table of his animations in the code that moves him.

        That code is a file of its own. It is found as the file holding the longest run of words
        that are addresses of his animations' headers, and the run is the table (on the first
        release 310 long: 45 rows of six, then the jumps and the pairs).

        Returns:
            (the file of that code, where the table is in it, [the headers' addresses]).

        Raises:
            BuildError: If no file holds such a run.
        """
        rom = self.rom
        heads = {struct.pack(">I", 0x04000000 + h) for h in self.animations}
        best = (0, None, 0)
        for i, (start, end, _, _) in enumerate(rom.files):
            if end - start < 4 * HANDS * len(MOVES) or i in {rom.program_file, self.animation_file}:
                continue
            if rom.packed_otherwise(i):
                continue
            d = rom.file(i)
            p = d.find(b"\x04")
            while 0 <= p < len(d) - 4:
                if p % 4 or d[p : p + 4] not in heads:
                    p = d.find(b"\x04", p + 1)
                    continue
                q = p
                while d[q : q + 4] in heads:
                    q += 4
                if q - p > best[0]:
                    best = (q - p, i, p)
                p = d.find(b"\x04", q)
        length, number, at = best
        if length < 4 * HANDS * len(MOVES):
            raise BuildError("the table of Link's animations in the code that moves him was not found")
        words = struct.unpack_from(f">{length // 4}I", rom.file(number), at)
        return number, at, [w & 0xFFFFFF for w in words]

    def _table_rows(self) -> int:
        """How many rows of six the table has, or 0 if it is not laid out as rows, jumps and pairs."""
        table = self.move_table()[2]
        count = (len(table) - 2 * IDLE_PAIRS - 3 * JUMPS) // HANDS
        return count if len(table) == HANDS * count + 3 * JUMPS + 2 * IDLE_PAIRS else 0

    @_found_once
    def moves(self) -> dict[str, int]:
        """
        His stand, walk and run, each with empty hands and with the sword and shield out, the look
        about him he does left standing, and his backflip.

        Checked two ways before they are believed: every row repeats its first column in its fifth
        and sixth (the other things he can have in his hands that change nothing), and the three
        move as their names say: his feet stay put in the first, travel in the second and travel
        further in a shorter cycle in the third.

        Returns:
            {what he does: its animation's header}.

        Raises:
            BuildError: If the table is not laid out that way.
        """
        table = self.move_table()[2]
        rows = [table[HANDS * k : HANDS * k + HANDS] for k in range(len(MOVES))]
        if any(row[0] != row[4] or row[0] != row[5] for row in rows):
            raise BuildError(f"the table of Link's animations is not in rows of {HANDS}")
        found = {name: row[0] for name, row in zip(MOVES, rows, strict=True)}
        left, right = PARTS.index("left_foot"), PARTS.index("right_foot")

        def steps(columns: dict[str, int]) -> bool:
            travel, length = {}, {}
            for name, header in columns.items():
                a = Animation(self, header)
                gaps = []
                for n in range(a.frames):
                    pose = self.pose(a.rotations(float(n)), a.root(float(n)))
                    gaps.append(pose[left][1][2] - pose[right][1][2])
                travel[name], length[name] = max(gaps) - min(gaps), a.frames
            return (
                travel["wait"] < 200
                and 1000 < travel["walk"] < travel["run"]
                and length["run"] < length["walk"] < length["wait"]
            )

        # (on the first release, in his own units, about 6,400 tall: his feet travel 2, 3197 and 4865)
        if not steps(found):
            raise BuildError("the first rows of the table of Link's animations are not a stand, a walk and a run")
        # The same three with his sword and shield out: the second column. Each must be another
        # animation than the one for empty hands, and they must stand, step and run as their rows do.
        drawn = {name: row[1] for name, row in zip(MOVES, rows, strict=True)}
        if all(drawn[name] != found[name] for name in MOVES) and steps(drawn):
            found.update({name + ARMED: header for name, header in drawn.items()})
        # Looking about him: the first of the pairs the table ends with, its second column. It must
        # begin and end in his stand (the other column, for a sword in hand, does not), and in it he
        # must turn his head to look behind him. (On the first release the limb furthest from the
        # stand's first frame is 14 degrees off it at both ends, 87 in the other column, and at its
        # furthest his head faces 127 degrees from straight ahead.)
        if len(table) >= 2 * IDLE_PAIRS + HANDS * len(MOVES):
            armed, look = table[len(table) - 2 * IDLE_PAIRS :][:2]
            stand = Animation(self, found["wait"]).frame(0)

            def off(a: Animation, n: int) -> float:
                frame = a.frame(n)
                turned = max(abs((frame[k] - stand[k] + 0x8000) % 0x10000 - 0x8000) for k in range(3, len(stand) - 1))
                return turned * 360.0 / 65536

            a, other = Animation(self, look), Animation(self, armed)
            head = PARTS.index("head")
            back = min(self.pose(a.rotations(float(n)), a.root(float(n)))[head][0][2][0] for n in range(a.frames))
            if off(a, 0) < 30.0 and off(a, a.frames - 1) < 30.0 and off(other, 0) > 45.0 and back < -0.5:
                found["look"] = look
                # the other column is the same look with his sword and shield out, from that stand
                if "wait" + ARMED in found:
                    stand = Animation(self, found["wait" + ARMED]).frame(0)
                    back = min(
                        self.pose(other.rotations(float(n)), other.root(float(n)))[head][0][2][0]
                        for n in range(other.frames)
                    )
                    if off(other, 0) < 30.0 and off(other, other.frames - 1) < 30.0 and back < -0.5:
                        found["look" + ARMED] = armed
        # His jump backwards: the third of the four jumps before the pairs. Of the four it must be the
        # only one to turn him upside down, it must turn him backwards, and its landing must begin
        # where it ends. (On the first release his trunk points straight down at its lowest and turns
        # 286 degrees backwards; the landing's first frame is the jump's last.)
        if len(table) >= 2 * IDLE_PAIRS + 3 * JUMPS + HANDS * len(MOVES):
            jumps = table[len(table) - 2 * IDLE_PAIRS - 3 * JUMPS :][: 3 * JUMPS]
            lowest = [self.trunk(Animation(self, jumps[3 * k])) for k in range(JUMPS)]
            flip, land = Animation(self, jumps[6]), Animation(self, jumps[7])
            if (
                [low < 0.0 for low, _ in lowest] == [False, False, True, False]
                and lowest[2][0] < -0.8
                and lowest[2][1] < -180.0
                and flip.frame(flip.frames - 1) == land.frame(0)
            ):
                found["backflip"], found["backflip_land"] = jumps[6], jumps[7]
        return found

    @_found_once
    def dodges(self) -> dict[str, int]:
        """
        His roll: `roll`, and `roll_armed` for the sword and shield out, when that is another
        animation.

        The roll is the row `ROLL_ROW` of the rows of six. Of all the rows it must be the only one
        that turns him over, in its first column and its second, and each must end in the stand it
        belongs to. (On the first release it alone takes his trunk below upright at all, to -0.97,
        where the next lowest row has 0.01; it turns him 375 degrees forwards, and its last frame is
        the stand's first, to the degree.) A name that is not what it says is left out.

        Returns:
            {what he does: its animation's header}.
        """
        table, moves, found = self.move_table()[2], self.moves(), {}
        count = self._table_rows()
        if count > ROLL_ROW:
            trunks = [[self.trunk(Animation(self, table[HANDS * k + c])) for c in (0, 1)] for k in range(count)]
            over = [k for k, columns in enumerate(trunks) if any(low < -0.5 for low, _ in columns)]
            if over == [ROLL_ROW] and all(low < -0.8 and turned > 300.0 for low, turned in trunks[ROLL_ROW]):
                for name, header in (("", table[HANDS * ROLL_ROW]), (ARMED, table[HANDS * ROLL_ROW + 1])):
                    a = Animation(self, header)
                    if (
                        "wait" + name in moves
                        and apart(a.turns(a.frames - 1.0), Animation(self, moves["wait" + name]).turns(0.0)) < 15.0
                    ):
                        found["roll" + name] = header
                if found.get("roll") == found.get("roll" + ARMED):
                    found.pop("roll" + ARMED, None)
        return found

    def code(self) -> tuple[tuple[int, ...], int]:
        """
        The code that moves him.

        Where it sits in memory is read from the program's own entry for the file: its place in the
        ROM, then the addresses it is loaded between.

        Returns:
            Its words, and where it sits in memory.

        Raises:
            BuildError: If the program has no such entry.
        """
        rom = self.rom
        number = self.move_table()[0]
        code = rom.file(number)
        at = rom.program.find(struct.pack(">2I", *rom.files[number][:2]))
        low, high = struct.unpack_from(">2I", rom.program, at + 8) if at >= 0 else (0, 0)
        if not low or high - low < len(code):
            raise BuildError("where the code that moves Link is loaded was not found in the program")
        return mips.words_of(code), low

    def _roll_routine(self) -> tuple[tuple[int, ...], int, int, int, int]:
        """The routine that plays his roll: the one that takes an animation from the roll's row of the table."""
        words, low = self.code()
        row = low + self.move_table()[1] + 4 * HANDS * ROLL_ROW
        found = mips.loads(words, row, row + 4 * HANDS - 4)
        if len(found) != 1:
            raise BuildError(
                f"the routine that plays Link's roll was not found ({len(found)} take an animation from its row)",
            )
        first, end = mips.routine(words, found[0][0])
        return words, low, found[0][0], first, end

    @_found_once
    def roll_speed(self) -> float:
        """
        How fast his roll is played.

        Read from the code that moves him, by what the code does: the one routine that takes an
        animation from the roll's row of the table builds one number, multiplies it by one it keeps
        beside its tables, and hands the product to the call that plays the animation. On the first
        release the number built is 1.25 and the one kept is 1.

        Returns:
            Frames of its animation to a frame of his game.

        Raises:
            BuildError: If the routine does not do that, or the speed is not believable.
        """
        words, low, load, first, end = self._roll_routine()
        built = mips.floats(words, first, end)
        kept = [
            (words[i + k] >> 16 & 31, ((words[i] & 0xFFFF) << 16) + ((words[i + k] & 0xFFFF ^ 0x8000) - 0x8000))
            for i in range(first, end - 2)
            if words[i] >> 16 == 0x3C01
            for k in (1, 2)
            if words[i + k] >> 26 == 0x31 and words[i + k] >> 21 & 31 == 1
        ]
        calls = [i for i in range(load, end) if words[i] >> 26 == 3]
        if len(built) != 1 or len(kept) != 1 or not calls or not low <= kept[0][1] <= low + 4 * len(words) - 4:
            raise BuildError("the routine that plays Link's roll does not set its speed the way expected")
        (reg, number), (other, where) = next(iter(built.values())), kept[0]
        # the two multiplied together, and the product handed on before the call
        products = [
            w >> 6 & 31
            for w in words[first : calls[0]]
            if w >> 26 == 0x11 and w >> 21 & 31 == 16 and w & 0x3F == 2 and {w >> 11 & 31, w >> 16 & 31} == {reg, other}
        ]
        if len(products) != 1 or not any(
            w >> 21 == 0x220 and w >> 11 & 31 == products[0] for w in words[first : calls[0] + 2]
        ):
            raise BuildError("the routine that plays Link's roll does not hand on the speed it works out")
        speed = number * mips.to_float(words[(where - low) // 4])
        if not 0.5 <= speed <= 3.0:
            raise BuildError(f"Link's roll would be played at {speed:g} times its speed, which is not believable")
        return speed

    def roll_voice(self) -> int:
        """
        The number of his voice that his own roll asks for.

        The routine that plays his roll hands his game one routine of this file to run him by while
        he rolls, and that routine asks his voice for one number, in one place (by the public
        documentation, where the roll ends against a wall). On the first release: 0x6804.

        Returns:
            The number.

        Raises:
            BuildError: If that is not what the code does.
        """
        words, low, _, first, end = self._roll_routine()
        # the routines of this file whose addresses it builds: an upper half, and a lower half added to it
        upper, built = {}, []
        for i in range(first, end):
            w = words[i]
            if w >> 26 == 0x0F:
                upper[w >> 16 & 31] = (w & 0xFFFF) << 16
            elif w >> 26 == 0x09 and (w >> 21 & 31) in upper:
                address = upper[w >> 21 & 31] + ((w & 0xFFFF ^ 0x8000) - 0x8000)
                k = (address - low) // 4
                if low <= address < low + 4 * len(words) and not address & 3 and mips.routine(words, k)[0] == k:
                    built.append(k)
        asked = []
        for k in built:
            a, b = mips.routine(words, k)
            # ("the number into the second argument", as his voice is asked for everywhere: 0x68NN is his voice's)
            asked += [
                words[j] & 0xFFFF for j in range(a, b) if words[j] >> 16 == 0x2405 and (words[j] & 0xFFFF) >> 8 == 0x68
            ]
        if len(built) != 1 or len(asked) != 1:
            raise BuildError("the routine that runs Link's roll does not ask his voice for one number")
        return asked[0]

    def _over(self, a: Animation, n: float, tall: float) -> tuple[float, float]:
        """How far above the joint of his head his lower wrist is, and his higher, as parts of `tall`."""
        head, wrists = PARTS.index("head"), (PARTS.index("left_hand"), PARTS.index("right_hand"))
        pose = self.pose(a.turns(float(n)), a.root(float(n)))
        ys = [(pose[w][1][1] - pose[head][1][1]) / tall for w in wrists]
        return min(ys), max(ys)

    @_found_once
    def carries(self) -> dict[str, int]:
        """
        A thing held over his head, as he holds a bomb.

        `lift` (he raises it, from empty hands), `hold` (he stands with it there), `throw` and
        `throw_armed` (he throws it, and ends in his stand with empty hands or with the sword and
        shield out), and `put_armed` (he sets it down in front of him, and ends in his stand with
        the sword and shield out). `lift` is the first column of the row `CARRY_ROW`, the throws
        the first two of `THROW_ROW`, and setting it down the second of `PUT_ROW`. His wrists must
        begin the lift below the joint of his head and end it above, and begin a throw above and end
        it below, in the stand the column belongs to. `hold` is not in the table: it is, among the
        animations of a second or more that come round and in which both his wrists stay as far
        above his head as the lift leaves them and hardly move, the one that begins nearest where
        the lift ends. Heights are measured against how high the joint of his head is as he
        stands. A name that is not what it says is left out.

        Returns:
            {what he does: its animation's header}.
        """
        table, moves, found = self.move_table()[2], self.moves(), {}
        head = PARTS.index("head")
        stand = Animation(self, moves["wait"])
        tall = self.pose(stand.turns(0.0), stand.root(0.0))[head][1][1]

        def bent(a: Animation) -> float:
            """How near upright (1) his trunk is on the first frame his wrists are both below his head's joint."""
            # It tells throwing a thing from setting it down: he is bent into a throw as his hands come down, and
            # brings them down before he bows to set a thing down.
            waist = PARTS.index("waist")
            for n in range(a.frames):
                if self._over(a, n, tall)[1] < 0.0:
                    pose = self.pose(a.turns(float(n)), a.root(float(n)))
                    v = [pose[head][1][k] - pose[waist][1][k] for k in range(3)]
                    return v[1] / math.sqrt(sum(c * c for c in v))
            return 1.0

        count = self._table_rows()
        if count > max(CARRY_ROW, THROW_ROW):
            lift = Animation(self, table[HANDS * CARRY_ROW])
            if self._over(lift, 0, tall)[1] < -0.15 and self._over(lift, lift.frames - 1, tall)[0] > 0.15:
                found["lift"] = table[HANDS * CARRY_ROW]
                up = self._over(lift, lift.frames - 1, tall)[0]
                end, held = lift.turns(lift.frames - 1.0), []
                for header in self.animations:
                    a = Animation(self, header)
                    if a.frames < 20 or header == found["lift"]:
                        continue
                    spans = [self._over(a, n, tall) for n in range(a.frames)]
                    lowest = min(lo for lo, _ in spans)
                    if not (lowest > 0.9 * up and max(hi for _, hi in spans) - lowest < 0.05):
                        continue
                    first = a.turns(0.0)
                    if apart(a.turns(a.frames - 1.0), first) > 2.0:
                        continue  # it does not come round
                    held.append((apart(first, end), header))
                held.sort()
                # (believed only if it is nearer the lift's end than any other by a margin: 10 degrees)
                if held and (len(held) == 1 or held[1][0] - held[0][0] > 10.0):
                    found["hold"] = held[0][1]
            for name, column in (("", 0), (ARMED, 1)):
                if "wait" + name not in moves:
                    continue
                a, home = Animation(self, table[HANDS * THROW_ROW + column]), Animation(self, moves["wait" + name])
                if (
                    self._over(a, 0, tall)[0] > 0.08
                    and self._over(a, a.frames - 1, tall)[1] < -0.1
                    and bent(a) < 0.8
                    and apart(a.turns(a.frames - 1.0), home.turns(0.0)) < 15.0
                ):
                    found["throw" + name] = table[HANDS * THROW_ROW + column]
            # Setting it down, with the sword and shield out: both wrists over his head as it begins,
            # his trunk bowed over his hands on the way, and that stand at its end.
            if count > PUT_ROW and "wait" + ARMED in moves:
                a, home = Animation(self, table[HANDS * PUT_ROW + 1]), Animation(self, moves["wait" + ARMED])
                a.loops = False
                if (
                    self._over(a, 0, tall)[0] > 0.15
                    and self._over(a, a.frames - 1, tall)[1] < -0.1
                    and self.trunk(a)[0] < 0.7
                    and bent(a) > 0.9
                    and apart(a.turns(a.frames - 1.0), home.turns(0.0)) < 15.0
                ):
                    found["put" + ARMED] = table[HANDS * PUT_ROW + 1]
        return found

    @_found_once
    def idles(self) -> dict[str, int]:
        """
        The things he does when left standing with the sword and shield out, each beginning and
        ending in that stand.

        `idle_glance` (he turns and looks down behind him), `idle_stretch` (both arms out and up over
        his head), `idle_boot` (he lifts a foot behind him and looks at it), `idle_shield` (he raises
        the shield and looks it over) and `idle_flourish` (he tries a swing of the sword). Each is the
        first column of the pair `IDLES` names, of the fourteen the table ends with, and is believed
        only if it is what its name says, by how high his wrists get and how far a foot is lifted,
        against how high the joint of his head is as he stands.

        Returns:
            {what he does: its animation's header}.
        """
        table, moves, found = self.move_table()[2], self.moves(), {}
        if "wait" + ARMED not in moves or not self._table_rows():
            return found
        home = Animation(self, moves["wait" + ARMED]).turns(0.0)
        stand = self.pose(home, Animation(self, moves["wait" + ARMED]).root(0.0))
        at = {
            name: PARTS.index(name) for name in ("waist", "head", "left_hand", "right_hand", "left_foot", "right_foot")
        }
        tall, hips = stand[at["head"]][1][1], stand[at["waist"]][1][1]

        def seen(header: int) -> tuple[Animation, tuple[float, float], float, float]:
            """(the animation; how high each wrist gets; how far a foot is lifted; how low his hips get)."""
            a = Animation(self, header)
            a.loops = False
            poses = [self.pose(a.turns(float(n)), a.root(float(n))) for n in range(a.frames)]

            def top(part: str) -> float:
                return max(p[at[part]][1][1] for p in poses)

            return (
                a,
                (top("left_hand") / tall, top("right_hand") / tall),
                max((top(foot) - stand[at[foot]][1][1]) / tall for foot in ("left_foot", "right_foot")),
                min(p[at["waist"]][1][1] for p in poses) / hips,
            )

        def is_what(name: str, wrists: tuple[float, float], foot: float) -> bool:
            """Whether how high his wrists get and how far a foot is lifted are what the thing he does needs."""
            return {
                "glance": max(wrists) < 0.9 and foot < 0.1,
                "stretch": min(wrists) > 1.2,
                "boot": foot > 0.12 and max(wrists) < 0.9,
                "shield": wrists[1] > 1.1 and wrists[0] < 0.8,
                "flourish": wrists[0] > 1.0 and wrists[1] < 0.9,
            }[name]

        pairs = table[len(table) - 2 * IDLE_PAIRS :]
        for name, pair in IDLES.items():
            a, wrists, foot, low = seen(pairs[2 * pair])
            if (
                a.frames >= 40
                and low > 0.9
                and is_what(name, wrists, foot)
                and apart(a.turns(0.0), home) < 30.0
                and apart(a.turns(a.frames - 1.0), home) < 30.0
            ):
                found["idle_" + name] = pairs[2 * pair]
        return found

    @_found_once
    def bows(self) -> dict[str, int]:
        """
        What he does with his bow.

        `bow_aim` (he stands with the bow held out and the string drawn to his cheek; it comes
        round), `bow_ready` (he raises the bow from his side and draws), `bow_loose` (the string hand
        flung back as the arrow leaves) and `bow_next` (from there he reaches forward and draws
        again). None is in the table (the code names them itself), so each is found among all his
        animations by what it is; heights are against how high the joint of his head is as he stands
        with the sword out. The aim is the one animation of 40 frames or more that comes round with
        his hips up, both wrists high and level and the bow hand well ahead of the other; raising it
        and taking the next arrow are the short ones that end in the aim, the first beginning with
        his hands low and the second with his left over his head; loosing is the very short one
        whose last frame is by far the nearest the first of taking the next arrow.

        Returns:
            {what he does: its animation's header}.
        """
        found, moves = {}, self.moves()
        if "wait" + ARMED not in moves:
            return found
        home = Animation(self, moves["wait" + ARMED])
        stand = self.pose(home.turns(0.0), home.root(0.0))
        at = {name: PARTS.index(name) for name in ("waist", "head", "left_hand", "right_hand")}
        tall, hips = stand[at["head"]][1][1], stand[at["waist"]][1][1]

        def hands(a: Animation, n: int) -> tuple[Vector, Vector, float]:
            pose = self.pose(a.turns(float(n)), a.root(float(n)))
            return pose[at["left_hand"]][1], pose[at["right_hand"]][1], pose[at["waist"]][1][1] / hips

        def aimed(a: Animation) -> bool:
            if a.frames < 40 or apart(a.turns(a.frames - 1.0), a.turns(0.0)) > 1.0:
                return False
            for n in range(a.frames):
                left, right, up = hands(a, n)
                if not (
                    up > 0.9
                    and min(left[1], right[1]) > 0.9 * tall
                    and abs(left[1] - right[1]) < 0.05 * tall
                    and right[2] - left[2] > 0.3 * tall
                ):
                    return False
            return True

        every = []
        for header in sorted(self.animations):
            a = Animation(self, header)
            a.loops = False
            every.append(a)
        aims = [a for a in every if aimed(a)]
        if len(aims) == 1:
            found["bow_aim"] = aims[0].header
            first = aims[0].turns(0.0)
            into = [a for a in every if a.frames <= 20 and apart(a.turns(a.frames - 1.0), first) < 15.0]
            ready = [a for a in into if max(hands(a, 0)[0][1], hands(a, 0)[1][1]) < 0.8 * tall and hands(a, 0)[2] > 0.9]
            after = [a for a in into if hands(a, 0)[0][1] >= tall]
            if len(ready) == 1:
                found["bow_ready"] = ready[0].header
            if len(after) == 1:
                found["bow_next"] = after[0].header
                begins = after[0].turns(0.0)
                short = sorted((apart(a.turns(a.frames - 1.0), begins), a.header) for a in every if a.frames <= 5)
                if len(short) > 1 and short[0][0] < 45.0 and short[1][0] > 2.0 * short[0][0]:
                    found["bow_loose"] = short[0][1]
        return found

    @_found_once
    def sits(self) -> tuple[dict[str, int], float | None]:
        """
        His sitting down on the ground: `sit_down` (he stands, and drops back onto his seat), `sit`
        (he sits with his legs out and his hands on the ground behind him; it comes round) and
        `sit_up` (he gets up).

        They are not in the table and nothing names them, so they are found by what they are, among
        all of his animations. `sit`: the one of a second or more that comes round with his hips
        under a third of their standing height and his trunk upright throughout. `sit_down`: the one
        that begins in his stand with empty hands and ends where `sit` begins; `sit_up` the one that
        begins there and ends in that stand. Each must be the only one.

        They are made for a shorter body, young Link's by every sign: in the standing first frame of
        `sit_down` his hips are at 0.640 of the height his own stand has them (as in fifteen more of
        his animations, two of them the young Link's ways through a door). The root's height in all
        three is to be divided by that share: with it he stands on the ground and sits on it.

        Returns:
            ({what he does: its animation's header}, that share), or ({}, None) unless all three are found.
        """
        moves, found, share = self.moves(), {}, None
        home = Animation(self, moves["wait"])
        stand, tall = home.turns(0.0), home.root(0.0)[1]
        every = [Animation(self, header) for header in self.animations]
        for a in every:
            a.loops = False
        stays = []
        for a in every:
            if a.frames < 20 or apart(a.turns(a.frames - 1.0), a.turns(0.0)) > 2.0:
                continue
            roots = [a.root(float(n))[1] / tall for n in range(a.frames)]
            if not (min(roots) > 0.1 and max(roots) < 0.34):
                continue
            if self.trunk(a)[0] > 0.7:
                stays.append(a)
        if len(stays) == 1:
            first = stays[0].turns(0.0)
            downs = [
                a
                for a in every
                if a is not stays[0]
                and apart(a.turns(0.0), stand) < 20.0
                and apart(a.turns(a.frames - 1.0), first) < 20.0
            ]
            ups = [
                a
                for a in every
                if a is not stays[0]
                and apart(a.turns(0.0), first) < 20.0
                and apart(a.turns(a.frames - 1.0), stand) < 20.0
            ]
            if len(downs) == 1 and len(ups) == 1:
                share = downs[0].root(0.0)[1] / tall
                if 0.5 < share < 0.8 and abs(ups[0].root(ups[0].frames - 1.0)[1] / tall - share) < 0.02:
                    found = {"sit_down": downs[0].header, "sit": stays[0].header, "sit_up": ups[0].header}
        return found, share if found else None

    def keys(self, pieces: tuple) -> Keys:
        """Several of his animations end to end as they are, nothing eased."""
        return Keys(self, pieces)

    def sequence(self, pieces: tuple) -> Sequence:
        """Several of his animations as one that comes round."""
        return Sequence(self, pieces)

    def trunk(self, a: Animation) -> tuple[float, float]:
        """
        How his trunk moves in an animation, taken from the joint of his waist to the joint of his head.

        Returns:
            How near upright it is at its lowest (1 upright, -1 upside down), and the degrees it turns
            over in all, forwards positive.
        """
        waist, head = PARTS.index("waist"), PARTS.index("head")
        low, turned, before = 1.0, 0.0, None
        for n in range(a.frames):
            pose = self.pose(a.rotations(float(n)), a.root(float(n)))
            v = tuple(pose[head][1][k] - pose[waist][1][k] for k in range(3))
            low = min(low, v[1] / (math.sqrt(sum(c * c for c in v)) or 1.0))
            angle = math.atan2(v[2], v[1])
            if before is not None:
                turned += (angle - before + math.pi) % (2.0 * math.pi) - math.pi
            before = angle
        return low, math.degrees(turned)

    # -- the swings of his sword ---------------------------------------------------------------------

    @_found_once
    def swing_table(self) -> tuple[int, int, list[tuple[int, int, int, int, int]]]:
        """
        The table of his sword's swings in the code that moves him: the one run, sixteen or more
        long, of three addresses of his animations, two bytes and two zeros.

        Returns:
            (the file, where the table is in it, [(swing, back, back with something targeted,
            first frame, last frame)]).

        Raises:
            BuildError: If there is not one such run.
        """
        number = self.move_table()[0]
        d = self.rom.file(number)
        heads = {0x04000000 + h for h in self.animations}
        runs, p = [], 0
        while p <= len(d) - SWING_BYTES:
            q, rows = p, []
            while q <= len(d) - SWING_BYTES:
                a, b, c, first, last, zero = struct.unpack_from(">3IBBH", d, q)
                if not (a in heads and b in heads and c in heads) or zero:
                    break
                rows.append((a & 0xFFFFFF, b & 0xFFFFFF, c & 0xFFFFFF, first, last))
                q += SWING_BYTES
            if len(rows) >= 4 * len(SWINGS):
                runs.append((p, rows))
            p = q if rows else p + 4
        if len(runs) != 1:
            raise BuildError(f"the table of Link's swings was not found in the code that moves him ({len(runs)} fit)")
        return number, runs[0][0], runs[0][1]

    def blade(self) -> tuple[int, Vector]:
        """
        Where the blade's point is: the tip of whatever his left hand holds, in the hand's own space.

        Returns:
            (the left hand's place among the parts, the point furthest from its joint).
        """
        hand = PARTS.index("left_hand")
        rot, pos = self.pose()[hand]
        tip = max(
            (
                rotate(transpose(rot), tuple(c[0][k] - pos[k] for k in range(3)))
                for _, _, tri in self.skin()
                for c in tri
                if c[3] == "left_hand"
            ),
            key=lambda v: sum(x * x for x in v),
        )
        return hand, tip

    @_found_once
    def swings(self) -> dict[str, tuple[int, int, int, int]]:
        """
        His swings with one hand: each of `SWINGS`, and each again with `_combo` after it for the
        later blow of a run.

        Believed only if they are what the names say: each comes back to his stand with the sword
        out, each later blow is longer than the first of its kind, and, by how the blade's tip is
        moving as it passes furthest ahead of him, `forward` comes down, `right` goes across towards
        his left and `left` across the other way.

        Returns:
            {kind: (swing, back, first frame, last frame)}: the swing and the way back are headers of
            animations, and the frames are those of the swing on which the blade can hit.

        Raises:
            BuildError: If they are not.
        """
        rows = self.swing_table()[2]
        found = {}
        for k, kind in enumerate(SWINGS):
            found[kind] = rows[4 * k][:2] + rows[4 * k][3:]
            found[kind + "_combo"] = rows[4 * k + 2][:2] + rows[4 * k + 2][3:]
        stand = self.moves().get("wait" + ARMED)
        if stand is None:
            raise BuildError("Link's stand with the sword out was not found, so his swings cannot be checked")
        hand, tip = self.blade()

        def path(kind: str) -> list[Vector]:
            a = Animation(self, found[kind][0])
            out = []
            for n in range(found[kind][2], found[kind][3] + 1):
                rot, pos = self.pose(a.rotations(float(n)), a.root(float(n)))[hand]
                moved = rotate(rot, tip)
                out.append(tuple(pos[i] + moved[i] for i in range(3)))
            return out

        heights = [c[0][1] for _, _, tri in self.skin() for c in tri]
        tall = max(heights) - min(heights)
        for kind, (swing, back, first, last) in found.items():
            a, b, home = Animation(self, swing), Animation(self, back), Animation(self, stand)
            if not 0 <= first < last <= a.frames - 1:
                raise BuildError(f"Link's {kind} swing can hit from frame {first} to {last} of {a.frames}")
            if apart(b.turns(float(b.frames - 1)), home.turns(0.0)) > 15.0:
                raise BuildError(f"Link's {kind} swing does not come back to his stand with the sword out")
        for kind in SWINGS:
            if Animation(self, found[kind + "_combo"][0]).frames <= Animation(self, found[kind][0]).frames:
                raise BuildError(f"the later blow of Link's {kind} swing is not the longer")

        def passing(kind: str) -> tuple[float, float]:
            """How the blade's tip moves as it passes furthest ahead of him: (across, up), as parts of his height."""
            pts = path(kind)
            k = max(range(len(pts)), key=lambda i: pts[i][2])
            a, b = pts[max(k - 1, 0)], pts[min(k + 1, len(pts) - 1)]
            return (b[0] - a[0]) / tall, (b[1] - a[1]) / tall

        (fx, fy), (rx, ry), (lx, ly) = (passing(kind) for kind in ("forward", "right", "left"))
        if not (fy < 0.0 and -fy > 2.0 * abs(fx) and rx > 2.0 * abs(ry) and lx < 0.0 and -lx > abs(ly)):
            raise BuildError(
                "the first swings of the table are not one straight down, one from his right and one from his left",
            )
        return found

    def swing(self, kind: str) -> Sequence:
        """
        A swing of his sword as one animation that plays once: the swing, then the animation that
        brings him back to his stand with the sword out, eased in as his game eases a new animation
        in. Where the two would carry him forward (the later blows of two kinds step into it, and
        his game moves him by that) he is kept where he stands: what he would have travelled by the
        end is taken out evenly over the swing.

        Returns:
            The two as one `Sequence`, its `window` the frames of the swing on which the blade can hit.
        """
        swing, back, first, last = self.spins()["spin"] if kind == "spin" else self.swings()[kind]
        home = Animation(self, self.moves()["wait" + ARMED]).frame(0)
        end = Animation(self, back)
        end = end.frame(end.frames - 1)
        seq = Sequence(self, (swing, back), once=True, travel=(end[0] - home[0], end[2] - home[2]))
        seq.window = (first, last)
        return seq

    # -- a charge of his sword, and the spin it ends in ----------------------------------------------

    def _turned(self, a: Animation, left: str, right: str) -> float:
        """How far round an animation turns the line from one limb's joint to another's, seen from above, in degrees."""
        i, j = PARTS.index(left), PARTS.index(right)
        total, before = 0.0, None
        for n in range(a.frames):
            pose = self.pose(a.rotations(float(n)), a.root(float(n)))
            angle = math.atan2(pose[i][1][0] - pose[j][1][0], pose[i][1][2] - pose[j][1][2])
            if before is not None:
                total += (angle - before + math.pi) % (2.0 * math.pi) - math.pi
            before = angle
        return math.degrees(total)

    @_found_once
    def spins(self) -> dict:
        """
        What a charge of his sword, and the spin it ends in, are made of, read from the code that
        moves him and believed only when each is what its name says.

        How each is read:

        * After the table of swings the code keeps six pairs of his animations (each first with one
          hand and then with two) and then two pairs of bytes that name four records of the table.
          Those four are the only ones that turn his hips round. The first of the first pair is the
          spin used; his hands are never near each other in it, where the record beside it is the
          same with both hands.
        * The routine that starts a swing sets a byte of his to a number, which the code common to
          all he does takes one from each frame; the routine that starts a charge does so when that
          byte reaches another number and the button is held. That routine plays the first pair
          from a frame of it, and sets the charge to nothing.
        * The routine he holds the stance by adds a number each frame (a call handed it and 1.0, its
          limit) and, once the charge reaches another, lets a letting go spin him; the routine of
          the letting go asks for the bigger spin past a third.

        Returns:
            `spin`: (the swing, the way back, first frame, last frame) of the spin with one hand, a
            record of `swing_table`; `records`: the four records the code names for a spin (the spin
            and the bigger spin, each with one hand and then two). `charge_in`: the animation he goes
            from a swing into the charge by, and `from`, the frame of it his code starts it on;
            `charge`: the stance he then holds, which comes round; `charge_out`: out of the stance
            without a spin, back to his stand with the sword out. `handover`: his game's frames from
            the start of a swing to the start of a charge, the button held throughout.
            `charge_start`: (first, end) of the routine that starts a charge, and `charge_field`,
            where in him the charge is kept. `step`: how much the charge grows each frame of the
            stance, from nothing; `spin_at`: what it must have reached for a letting go to spin him;
            `big_at`: and for the bigger spin. `ready`: the frames of the stance, as his game counts
            in single-precision floats, until a letting go spins him.

        Raises:
            BuildError: If any of that is not what the code does.
        """
        number, at, rows = self.swing_table()
        d = self.rom.file(number)
        heads = {0x04000000 + h for h in self.animations}
        pairs_at = at + SWING_BYTES * len(rows)
        pairs = []
        while len(pairs) < 2 * len(CHARGES):
            a, b = struct.unpack_from(">2I", d, pairs_at + 8 * len(pairs))
            if a not in heads or b not in heads:
                break
            pairs.append((a & 0xFFFFFF, b & 0xFFFFFF))
        if len(pairs) != len(CHARGES):
            raise BuildError(
                f"after the table of swings the code keeps {len(pairs)} pairs of animations, not {len(CHARGES)}",
            )
        names_at = pairs_at + 8 * len(pairs)
        named = struct.unpack_from(">8B", d, names_at)
        records = [named[0], named[1], named[4], named[5]]
        if (
            any(named[k] for k in (2, 3, 6, 7))
            or len(set(records)) != 4
            or not all(len(rows) - 4 <= r < len(rows) for r in records)
        ):
            raise BuildError(
                "after the pairs of a charge the code does not name four records at the end of the table of swings",
            )
        # the spins turn his hips round, and nothing else in the table does
        hips = [abs(self._turned(Animation(self, row[0]), "left_thigh", "right_thigh")) for row in rows]
        if not all(hips[r] > 300.0 for r in records) or any(
            turn > 150.0 for k, turn in enumerate(hips) if k not in records
        ):
            raise BuildError("the records the code names for a spin are not the ones that turn him round")
        swing, back, _, first, last = rows[records[0]]
        lh, rh = PARTS.index("left_hand"), PARTS.index("right_hand")
        a = Animation(self, swing)
        nearest = min(
            math.sqrt(sum((p[lh][1][k] - p[rh][1][k]) ** 2 for k in range(3)))
            for p in (self.pose(a.rotations(float(n)), a.root(float(n))) for n in range(a.frames))
        )
        heights = [c[0][1] for _, _, tri in self.skin() for c in tri]
        if nearest < 0.25 * (max(heights) - min(heights)):
            raise BuildError("the spin with one hand brings his hands close together: it is not one-handed")
        found = {"spin": (swing, back, first, last), "big_spin": rows[records[2]][:2] + rows[records[2]][3:]}
        found["records"] = tuple(records)
        found.update({name: pair[0] for name, pair in zip(CHARGES, pairs, strict=True)})

        # the code: where the pairs and the table are in memory, and the routines that read them
        words, low = self.code()
        table, pairs_mem, names_mem = low + at, low + pairs_at, low + names_at

        def jumps(f: int, e: int) -> list[int]:
            return [((w & 0x3FFFFFF) << 2 | low & 0xF0000000) for w in words[f:e] if w >> 26 == 3]

        def routine(address: int) -> tuple[int, int]:
            return mips.routine(words, (address - low) // 4)

        def reads(first_: int, last_: int, ops: tuple[int, ...] = (0x23,)) -> list[tuple[int, int]]:
            """The routines that load from between two addresses."""
            return sorted({mips.routine(words, j) for j, _ in mips.loads(words, first_, last_, ops)})

        def writes_at(v: int) -> bool:
            """Whether an instruction puts something in the assembler's register."""
            op = v >> 26
            if op == 0:
                return (v >> 11 & 31) == 1
            return (
                op in {0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x20, 0x21, 0x23, 0x24, 0x25}
                and v >> 16 & 31 == 1
            )

        def kept(f: int, e: int) -> list[tuple[int, int, float]]:
            """(instruction, float register, number) for each float a stretch loads from the code's own file."""
            out = []
            for i in range(f, e - 1):
                w = words[i]
                if w >> 16 == 0x3C01:
                    for j in range(i + 1, min(i + 10, e)):
                        v = words[j]
                        if v >> 26 == 0x31 and v >> 21 & 31 == 1:
                            address = ((w & 0xFFFF) << 16) + ((v & 0xFFFF ^ 0x8000) - 0x8000)
                            if low <= address < low + 4 * len(words):
                                out.append((j, v >> 16 & 31, mips.to_float(words[(address - low) // 4])))
                            break
                        if writes_at(v):
                            break
            return out

        def compared(f: int, e: int, field: int) -> list[float]:
            """The numbers a stretch loads from the code's file and compares with a float of his kept at `field`."""
            his = {v >> 16 & 31 for v in words[f:e] if v >> 26 == 0x31 and v & 0xFFFF == field}
            ours = {reg: number_ for _, reg, number_ in kept(f, e)}
            out = []
            for v in words[f:e]:
                if v >> 21 == 0x230 and v & 0x3F in {0x3C, 0x3E}:  # c.lt.s / c.le.s fs, ft
                    fs, ft = v >> 11 & 31, v >> 16 & 31
                    out += [ours[a] for a, b in ((fs, ft), (ft, fs)) if a in ours and b in his]
            return out

        def builds(f: int, e: int) -> set[int]:
            """The addresses a stretch builds in a register: an upper half, and a lower half added to it."""
            out = set()
            for i in range(f, e):
                w = words[i]
                if w >> 21 != 0x1E0:  # lui r, H
                    continue
                for j in range(i + 1, min(i + 10, e)):
                    v = words[j]
                    if v >> 26 == 0x0F and v >> 16 & 31 == w >> 16 & 31:
                        break
                    if v >> 26 == 9 and v >> 21 & 31 == w >> 16 & 31:
                        out.add(((w & 0xFFFF) << 16) + ((v & 0xFFFF ^ 0x8000) - 0x8000))
                        break
            return out

        # the routine that starts a charge from a swing: it reads the first two pairs, starts the first on a frame of it
        entries = reads(pairs_mem, pairs_mem + 15)
        if len(entries) != 1:
            raise BuildError(
                f"the routine that starts a charge was not found ({len(entries)} read the first two pairs)",
            )
        e_first, e_end = entries[0]
        found["charge_start"] = (e_first, e_end)  # (for `SpinRing`, which starts from it)
        frames_in = Animation(self, found["charge_in"]).frames
        started = [
            (i, n) for i, (_, n) in mips.floats(words, e_first, e_end).items() if 0.0 < n < frames_in and n == int(n)
        ]
        if len(started) != 1:
            raise BuildError("the routine that starts a charge does not start it on one frame of its animation")
        found["from"] = int(started[0][1])
        # it hands its work on to a routine that sets a float of his to nought: the charge, which begins at nothing
        zeroed = set()
        for target in jumps(e_first, e_end):
            f, e = routine(target)
            zero = {v >> 11 & 31 for v in words[f:e] if v & 0xFFFF07FF == 0x44800000}  # mtc1 $zero, fN
            zeroed |= {v & 0xFFFF for v in words[f:e] if v >> 26 == 0x39 and v >> 16 & 31 in zero}
        # the routine he holds the stance by: it reads the third pair, calls one that adds a number a frame to
        # that float (a call handed the number and 1.0, its limit), and compares the float with one number
        held = []
        for f, e in reads(pairs_mem + 16, pairs_mem + 23):
            for target in jumps(f, e):
                g, h = routine(target)
                body = words[g:h]
                for i in range(len(body) - 1):
                    w, v = body[i], body[i + 1]
                    # lui a2, H; ori a2, a2, L; lui a1, 1.0
                    if w >> 16 == 0x3C06 and v >> 16 == 0x34C6 and 0x3C053F80 in body:
                        fields = [x & 0xFFFF for x in body if x >> 26 == 9 and x >> 16 & 31 == 4]  # addiu a0, aX, field
                        step = mips.to_float((w & 0xFFFF) << 16 | v & 0xFFFF)
                        held += [
                            (f, e, step, fld, g + i)
                            for fld in fields
                            if fld in zeroed and len(compared(f, e, fld)) == 1
                        ]
        if len(set(held)) != 1:
            raise BuildError(
                "the routine Link holds a charge by was not found, or does not grow the charge as expected",
            )
        s_first, s_end, step, field, _ = held[0]
        found["charge_field"] = field  # (where in him the charge is kept)
        at_spin = compared(s_first, s_end, field)
        # the routine of letting go: it reads the bytes that name the spins, and compares the charge again
        lets = [
            r
            for r in reads(names_mem, names_mem + 7, ops=(0x24, 0x20))
            if r[0] in [(t - low) // 4 for t in jumps(s_first, s_end)]
        ]
        if len(lets) != 1:
            raise BuildError(f"the stance does not let a spin be had by one routine of letting go ({len(lets)})")
        at_big = compared(lets[0][0], lets[0][1], field)
        if len(at_big) != 1 or not 0.0 < step < at_spin[0] < at_big[0] <= 1.0:
            raise BuildError(
                f"a charge grows by {step:g}, spins at {at_spin} and spins bigger at {at_big}: not believable",
            )
        found.update(step=step, spin_at=at_spin[0], big_at=at_big[0])

        def f32(x: float) -> float:
            return struct.unpack("<f", struct.pack("<f", x))[0]

        charge, found["ready"] = 0.0, 0
        while charge < f32(found["spin_at"]):
            charge, found["ready"] = f32(charge + f32(step)), found["ready"] + 1
        # the hand-over: a byte of his the swing's routine sets (that routine builds the table's address), which the
        # code takes one off each frame and the routine that starts a charge waits to see at a number
        sets = []
        for f, e in mips.routines(words):
            if table not in builds(f, e):
                continue
            for i in range(f, e):
                w = words[i]
                if w >> 21 == 0x120 and 0 < w & 0xFFFF < 64:  # addiu t, zero, N ...
                    for v in words[i + 1 : min(i + 4, e)]:
                        if v >> 26 == 0x28 and v >> 16 & 31 == w >> 16 & 31:  # ... sb t, F(s)
                            sets.append((w & 0xFFFF, v & 0xFFFF, i))
                            break
        counted, waited = set(), []
        for f, e in mips.routines(words):
            body = words[f:e]
            for _, fld, _ in sets:
                for i, w in enumerate(body):
                    if w >> 26 != 0x20 or w & 0xFFFF != fld:  # lb r, F(s)
                        continue
                    got = w >> 16 & 31
                    after = body[i + 1 : i + 20]
                    # counted down: one taken off what was loaded, and that stored back
                    less = [
                        v >> 16 & 31 for v in after if v >> 26 == 9 and v >> 21 & 31 == got and v & 0xFFFF == 0xFFFF
                    ]
                    if any(v >> 26 == 0x28 and v & 0xFFFF == fld and v >> 16 & 31 in less for v in after):
                        counted.add(fld)
                    # waited on: compared with a number, in a routine that calls one that starts the charge
                    if (
                        after
                        and after[0] >> 21 == 0x120
                        and after[0] >> 16 & 31 == 1
                        and any(e_first in [(t - low) // 4 for t in jumps(*routine(c))] for c in jumps(f, e))
                    ):
                        waited.append((fld, after[0] & 0xFFFF))
        hand = sorted({(n - m, i) for n, fld, i in sets for f2, m in waited if f2 == fld and fld in counted})
        if len(hand) != 1 or not 1 <= hand[0][0] <= 30:
            raise BuildError("how long a swing is held before a charge was not read")
        found["handover"] = hand[0][0]

        # what each animation of it does
        stand = Animation(self, self.moves()["wait" + ARMED])
        stance, into, out = (Animation(self, found[n]) for n in ("charge", "charge_in", "charge_out"))
        spun = Animation(self, back)
        joins = {
            "the stance comes round": apart(stance.turns(stance.frames - 1.0), stance.turns(0.0)) < 5.0,
            "its way in ends in it": apart(into.turns(into.frames - 1.0), stance.turns(0.0)) < 20.0,
            "the way out begins in it": apart(out.turns(0.0), stance.turns(0.0)) < 20.0,
            "the way out ends in his stand": apart(out.turns(out.frames - 1.0), stand.turns(0.0)) < 15.0,
            "the spin ends in his stand": apart(spun.turns(spun.frames - 1.0), stand.turns(0.0)) < 15.0,
            "he is down in the stance": stance.root(0.0)[1] < 0.85 * stand.root(0.0)[1],
        }
        if not all(joins.values()):
            raise BuildError(
                "a charge of Link's sword is not what it should be: not {}".format(
                    ", ".join(k for k, v in joins.items() if not v),
                ),
            )
        return found

    # -- display lists -------------------------------------------------------------------------------

    def _segment(self, addr: int) -> tuple[bytes, int]:
        """(data, offset) for an address a list names."""
        seg, off = addr >> 24, addr & 0xFFFFFF
        if seg == SEG_OBJECT:
            return self.object, off
        if seg == SEG_KEEP:
            return self.keep, off
        if seg == SEG_EYES:
            return self.object, (self.eye_table[self.eyes] & 0xFFFFFF) + off
        if seg == SEG_MOUTH:
            return self.object, (self.mouth_table[self.mouth] & 0xFFFFFF) + off
        raise BuildError(f"a list names segment {seg:#x}, which this reader cannot follow")

    def _build(self) -> None:
        self.parts = [Part(name, self.parent[i], self.limbs[i]["offset"]) for i, name in enumerate(PARTS)]
        self.materials = {}
        self._rest_root = None
        for i, limb in enumerate(self.limbs):
            addr = self.chosen.get(PARTS[i], limb["near"])
            if addr:
                self._run(addr, _State(self.env), i, 0)
        # The other things a limb can be drawn as: each read as the limb's own list is, into a set of
        # parts of its own, with the materials shared.
        self.look_tris = {}
        kept = self.parts
        for name, kind in self.looks.items():
            limb = PARTS.index(LOOK_LIMBS[kind.split()[0]])
            self.parts = [Part(part.name, part.parent, part.offset) for part in kept]
            try:
                self._run(self.held(kind), _State(self.env), limb, 0)
                tris = [tri for part in self.parts for tri in part.tris]
            finally:
                self.parts = kept
            astray = sorted({PARTS[c[0]] for _, corners, _ in tris for c in corners} - {PARTS[limb]})
            if astray:
                # (a corner that followed another limb would stay in sight when the look is hidden)
                raise BuildError(f"the list of the kind {kind} has corners that follow his {', '.join(astray)}")
            self.look_tris[name] = (limb, tris)

    def add_prop(self, part: str, look: str, tris: list) -> int:
        """
        Add a thing that is not a limb of his, for a model that is to show it at times.

        It is a part of its own, `part`, which `pose` puts where his root is unless it is told
        otherwise, and on it a look, `look` (`skin(pose, looks=True)` hands it over under that name),
        which is `tris` as `read_list` gives them, in the part's own space.

        Returns:
            The part's place in a pose.

        Raises:
            BuildError: If he already has something of either name.
        """
        if part in [p.name for p in self.parts] or look in self.look_tris or look in PARTS:
            raise BuildError(f"Link already has a {part} or a {look}")
        index = len(self.parts)
        self.parts.append(Part(part, 0, (0, 0, 0)))
        self.look_tris[look] = (index, [(m, [(index, *c[1:]) for c in corners], both) for m, corners, both in tris])
        return index

    def add_part(self, part: str) -> int:
        """
        Add a part of its own, as `add_prop` does, with nothing of his game's drawn on it.

        Returns:
            The part's place in a pose.

        Raises:
            BuildError: If he already has something of that name.
        """
        if part in [p.name for p in self.parts] or part in self.look_tris or part in PARTS:
            raise BuildError(f"Link already has a {part}")
        self.parts.append(Part(part, 0, (0, 0, 0)))
        return len(self.parts) - 1

    def read_list(self, addr: int, limb: int) -> list:
        """
        The triangles of one list as it would be drawn for a limb, leaving what was read into
        `parts` as it was.

        Returns:
            [(material, corners, both sides?)].
        """
        kept = self.parts, self.materials
        self.parts = [Part(name, self.parent[i], self.limbs[i]["offset"]) for i, name in enumerate(PARTS)]
        self.materials = {}
        try:
            self._run(addr, _State(self.env), limb, 0)
            return [tri for part in self.parts for tri in part.tris]
        finally:
            self.parts, self.materials = kept

    def _run(self, addr: int, state: _State, limb: int, depth: int) -> int:
        """Follow a display list. `limb` is the limb whose matrix is set; returns it as left."""
        if depth > 8:
            raise BuildError(f"display lists nest too deep at {addr:#x}")
        if addr >> 24 == SEG_CULL:
            state.geometry = state.geometry & ~G_CULL_FRONT | G_CULL_BACK  # the game sets back-face culling here
            return limb
        d, p = self._segment(addr)
        while True:
            w0, w1 = struct.unpack_from(">II", d, p)
            op = w0 >> 24
            p += 8
            if op == 0xDF:  # end
                return limb
            if op == 0xDE:  # another list
                limb = self._run(w1, state, limb, depth + 1)
                if w0 >> 16 & 0xFF:
                    return limb
            elif op == 0xDA:  # a matrix
                if w1 >> 24 != SEG_MATRIX:
                    raise BuildError(f"a list at {addr:#x} sets a matrix of its own ({w1:#x}); not handled")
                limb = self.matrix_limb[(w1 & 0xFFFFFF) // 0x40]
            elif op == 0x01:  # load vertices
                n, end = w0 >> 12 & 0xFF, (w0 & 0xFF) >> 1
                vd, vp = self._segment(w1)
                for k in range(n):
                    x, y, z, _, s, t, a, b, c, alpha = struct.unpack_from(">3hH2h4B", vd, vp + k * 16)
                    state.verts[end - n + k] = (limb, (x, y, z), (s, t), (a, b, c, alpha))
            elif op in {0x05, 0x06, 0x07}:  # one triangle, two, or a quad as two
                self._triangle(state, w0 >> 16 & 0xFF, w0 >> 8 & 0xFF, w0 & 0xFF)
                if op != 0x05:
                    self._triangle(state, w1 >> 16 & 0xFF, w1 >> 8 & 0xFF, w1 & 0xFF)
            elif op == 0xD7:  # texturing: on or off, which tile, the scale
                state.tex_on = bool(w0 & 0xFF)
                state.tex_tile = w0 >> 8 & 7
                state.tex_scale = tuple(1.0 if v == 0xFFFF else v / 65536.0 for v in (w1 >> 16, w1 & 0xFFFF))
            elif op == 0xD9:  # geometry mode: clear, then set
                state.geometry = state.geometry & w0 | w1
            elif op == 0xFD:  # the image named for the next load
                state.timg = (w1, w0 >> 21 & 7, w0 >> 19 & 3)
            elif op == 0xF5:
                state.tiles[w1 >> 24 & 7].set_tile(w0, w1)
            elif op == 0xF2:
                state.tiles[w1 >> 24 & 7].set_size(w0, w1)
            elif op in {0xF3, 0xF4}:  # load the named image where a tile points
                state.loaded[state.tiles[w1 >> 24 & 7].tmem] = state.timg[0]
            elif op == 0xF0:  # load a palette
                state.tlut = state.timg[0]
            elif op == 0xFC:
                state.combine = (w0, w1)
            elif op == 0xFA:
                state.prim = (w1 >> 24, w1 >> 16 & 255, w1 >> 8 & 255, w1 & 255)
            elif op == 0xFB:
                state.env = (w1 >> 24, w1 >> 16 & 255, w1 >> 8 & 255, w1 & 255)
            elif op == 0xE2:  # the low half of the modes: how a pixel is blended in
                shift = 32 - (w0 >> 8 & 0xFF) - ((w0 & 0xFF) + 1)
                mask = ((1 << ((w0 & 0xFF) + 1)) - 1) << shift
                state.mode = state.mode & ~mask | w1 & mask
            elif op == 0xE3:  # the high half: here only one or two cycles matters
                shift = 32 - (w0 >> 8 & 0xFF) - ((w0 & 0xFF) + 1)
                if shift <= 20 < shift + (w0 & 0xFF) + 1:
                    state.two_cycle = (w1 >> 20 & 3) == 1
            elif op not in QUIET:
                self.unknown[op] = self.unknown.get(op, 0) + 1

    def _triangle(self, state: _State, a: int, b: int, c: int) -> None:
        corners = [state.verts[i >> 1] for i in (a, b, c)]
        material = self._material(state)
        lit = bool(state.geometry & G_LIGHTING)
        both = not state.geometry & (G_CULL_BACK | G_CULL_FRONT)
        if state.geometry & G_CULL_FRONT and not state.geometry & G_CULL_BACK:
            corners.reverse()
        if material.texture:
            tile = material.tile
            for _, _, (s, t), _ in corners:
                u = s * material.scale[0] / 32.0 - (tile.uls >> 2)
                v = t * material.scale[1] / 32.0 - (tile.ult >> 2)
                r = material.range
                if r is None:
                    material.range = [math.floor(u), math.floor(u) + 1, math.floor(v), math.floor(v) + 1]
                else:
                    r[0], r[1] = min(r[0], math.floor(u)), max(r[1], math.floor(u) + 1)
                    r[2], r[3] = min(r[2], math.floor(v)), max(r[3], math.floor(v) + 1)
        # the triangle is kept with the limb most of its corners follow; each corner keeps its own
        limbs = [corner[0] for corner in corners]
        home = max(set(limbs), key=limbs.count)
        self.parts[home].tris.append(
            (
                material,
                [(limb, pos, st, col if not lit else None, col if lit else None) for limb, pos, st, col in corners],
                both,
            ),
        )

    def _material(self, state: _State) -> Material:
        tile = state.tiles[state.tex_tile]
        texgen = bool(state.geometry & G_TEXTURE_GEN)
        if state.tex_on:
            image = state.loaded.get(tile.tmem, 0)
            tkey = (image, state.tlut if tile.fmt == 2 else 0, *tile.key(), *state.tex_scale)
        else:
            image, tkey = 0, ()
        key = (tkey, state.combine, state.prim, state.env, state.mode & 0x5000, texgen, state.two_cycle)
        if key not in self.materials:
            texture = tlut = None
            copy = Tile()
            copy.__dict__.update(tile.__dict__)
            if image:
                data, at = self._segment(image)
                texture = (data, at, image)
                if tile.fmt == 2:
                    tlut = self._segment(state.tlut)
            self.materials[key] = Material(
                key,
                texture,
                copy,
                tlut,
                combiner(*state.combine),
                state.prim,
                state.env,
                state.mode,
                state.tex_scale,
                lit=bool(state.geometry & G_LIGHTING),
                texgen=texgen,
                two_cycle=state.two_cycle,
            )
        return self.materials[key]

    # -- what the builders ask for -------------------------------------------------------------------

    def image(self, material: Material) -> Image:
        """
        A material's picture: the stretch of its texture that its triangles reach, unrolled
        (repeats and mirrorings laid out side by side) and through the combiner. One texel of the
        texture is one pixel here.

        The picture is opaque unless the material is a cut-out. Then its alpha is the texture's, and
        the see-through pixels are given the colour of the nearest ones that show.

        Returns:
            (width, height, rows of (r, g, b, a)).
        """
        if not material.texture or material.texgen or material.range is None:
            flat = material.colour((255, 255, 255, 255)) if not material.texture else self._average(material)
            return 8, 8, [[(*flat[:3], 255)] * 8 for _ in range(8)]
        s0, s1, t0, t1 = material.range
        cache = {}

        def one(s: int, t: int) -> Pixel:
            texel = material.texel(s, t)
            if texel not in cache:
                cache[texel] = material.colour(texel)
            return cache[texel]

        rows = [[one(s, t) for s in range(s0, s1 + 1)] for t in range(t0, t1 + 1)]
        rows = bleed(rows) if material.cutout else [[(*px[:3], 255) for px in row] for row in rows]
        return s1 - s0 + 1, t1 - t0 + 1, rows

    def _average(self, material: Material) -> Pixel:
        w, h, rows = material.stored()
        total = [0, 0, 0]
        for row in rows:
            for px in row:
                c = material.colour(px)
                for i in range(3):
                    total[i] += c[i]
        return (*(v // (w * h) for v in total), 255)

    def pose(self, rotations: dict | None = None, root: Vector | None = None) -> Pose:
        """
        Him posed in model space.

        `rotations` maps part names to angles in radians about X, Y and Z, or to a rotation as a
        matrix (parts left out are not turned); `root` is where the root sits. With nothing given
        this is the rest pose. A thing hung on him (`add_prop`, `add_part`) lies where his root is in
        the rest pose, turned no way: no animation of his moves it.

        Returns:
            Every part's (rotation, position).
        """
        if rotations is None:
            rotations = {k: tuple(math.radians(a) for a in v) for k, v in REST.items()}
            root = self.rest_root() if root is None else root
        out = []
        for i, part in enumerate(self.parts):
            if i >= len(PARTS):
                out.append((IDENTITY, tuple(self._rest_root or (0.0, 0.0, 0.0))))
                continue
            rot = rotations.get(part.name, (0.0, 0.0, 0.0))
            rot = euler_matrix(*rot) if len(rot) == 3 and not isinstance(rot[0], (list, tuple)) else rot
            if part.parent is None:
                out.append((rot, tuple(float(c) for c in (root if root is not None else part.offset))))
            else:
                prot, ppos = out[part.parent]
                moved = rotate(prot, part.offset)
                out.append((matmul(prot, rot), tuple(ppos[k] + moved[k] for k in range(3))))
        return out

    def rest_root(self) -> Vector:
        """Where the root sits in the rest pose: over the middle, his soles on the ground."""
        if self._rest_root is None:
            rest = {k: tuple(math.radians(a) for a in v) for k, v in REST.items()}
            low = min(c[0][1] for _, _, tri in self.skin(self.pose(rest, (0.0, 0.0, 0.0))) for c in tri)
            self._rest_root = (0.0, -low, 0.0)
        return self._rest_root

    def triangles(self, pose: Pose | None = None) -> list[tuple[Material, str, list[Vertex]]]:
        """Every triangle in model space, as `skin` gives them without the part each corner follows."""
        return [(material, part, [c[:3] for c in tri]) for material, part, tri in self.skin(pose)]

    def skin(self, pose: Pose | None = None, *, looks: bool = False) -> list[SkinTriangle]:
        """
        Every triangle in model space.

        u and v are in widths and heights of the material's `image`, v downwards. A texel of the
        game's sits at a whole coordinate, so half a texel is added to put it at a pixel's middle. A
        triangle drawn from both sides in the game comes twice, the second time turned over. With
        `looks`, for a model that shows one of several things on a limb at a time: a limb that has
        looks hands over nothing under its own name, and each of its looks comes under the look's
        name, its corners following that name.

        Returns:
            (material, part, [(position, normal, (u, v), the part the corner follows)] * 3) of each.
        """
        pose = pose or self.pose()
        out = []
        groups = [(part.name, {}, part.tris) for part in self.parts]
        if looks:
            hosts = {limb for limb, _ in self.look_tris.values()}
            groups = [g for i, g in enumerate(groups) if i not in hosts]
            groups += [(name, {limb: name}, tris) for name, (limb, tris) in self.look_tris.items()]
        for part_name, renamed, tris in groups:
            for material, corners, both in tris:
                if material.texture and not material.texgen:
                    s0, s1, t0, t1 = material.range
                    w, h, tile = s1 - s0 + 1, t1 - t0 + 1, material.tile
                tri = []
                for limb, p, (s, t), _, normal in corners:
                    rot, pos = pose[limb]
                    q = rotate(rot, p)
                    if normal is not None:
                        n = rotate(rot, tuple(c - 256 if c > 127 else c for c in normal[:3]))
                        length = math.sqrt(sum(c * c for c in n)) or 1.0
                        n = tuple(c / length for c in n)
                    else:
                        n = None
                    if material.texture and not material.texgen:
                        uv = (
                            (s * material.scale[0] / 32.0 - (tile.uls >> 2) - s0 + 0.5) / w,
                            (t * material.scale[1] / 32.0 - (tile.ult >> 2) - t0 + 0.5) / h,
                        )
                    else:
                        uv = (0.5, 0.5)
                    name = renamed[limb] if limb in renamed else PARTS[limb]
                    tri.append((tuple(q[k] + pos[k] for k in range(3)), n, uv, name))
                if any(c[1] is None for c in tri):  # unlit: the face's own normal
                    a, b, c = (t[0] for t in tri)
                    e1, e2 = tuple(b[k] - a[k] for k in range(3)), tuple(c[k] - a[k] for k in range(3))
                    n = (e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0])
                    length = math.sqrt(sum(v * v for v in n)) or 1.0
                    n = tuple(v / length for v in n)
                    tri = [(p, n, uv, name) for p, _, uv, name in tri]
                out.append((material, part_name, tri))
                if both:
                    out.append(
                        (material, part_name, [(p, tuple(-v for v in n), uv, name) for p, n, uv, name in tri[::-1]]),
                    )
        return out
