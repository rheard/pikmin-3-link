"""
Things of Link's game that are actors of their own: the arrow his bow shoots, the bomb he throws,
and the ring of light his spin throws out round him. Each is found through the game's table of
actors and believed only if it is what it should be. And the trail his sword leaves, which is an
effect of his game's rather than an actor, read from the code that makes and draws it.

The actors' numbers (`ARROW`, `BOMB`) are the public documentation's; the ring's is read from his
code, which makes it.
"""

from __future__ import annotations

import math
import struct

from typing import TYPE_CHECKING

from link.errors import BuildError
from link.oot import mips
from link.oot.graphics import Image, Rows, Tile, address, bleed, combine, combiner, decode_texture
from link.oot.rom import KEEP

if TYPE_CHECKING:
    from link.geometry import Vector
    from link.oot.link import Link

BOMB = 0x10
ARROW = 0x16


class Arrow:
    """
    The arrow his bow shoots, as his game draws it.

    Its code names, in the object every scene shares, one skeleton whose limbs hang in a line and
    have one list between them, and that list is a long thin shape: a shaft, a head at one end and,
    at the other, feathers cut to shape by their texture and drawn from both sides. It is handed on
    as the list has it, lying along its own X: `tris` as `Link.read_list` gives them; `tail`, where
    along X the shaft ends at the feathers' end (the nock); `tip`, where the head's point is; and
    `length` from one to the other, in the units Link's own limbs are in.
    """

    def __init__(self, link: Link) -> None:
        keep = link.keep
        actor = link.rom.actor(ARROW)
        if actor.number != ARROW or actor.held_in != KEEP:
            raise BuildError(f"actor {ARROW:#x} is not drawn from the object every scene shares: not the arrow")
        named = sorted(v for v in mips.constants(actor.code)[1] if v >> 24 == 0x04 and (v & 0xFFFFFF) + 8 <= len(keep))
        found = []
        for v in named:
            limbs, count = struct.unpack_from(">IB", keep, v & 0xFFFFFF)
            if limbs >> 24 != 0x04 or not 0 < count <= 8 or (limbs & 0xFFFFFF) + 4 * count > len(keep):
                continue
            rows = struct.unpack_from(f">{count}I", keep, limbs & 0xFFFFFF)
            if any(p >> 24 != 0x04 or (p & 0xFFFFFF) + 12 > len(keep) for p in rows):
                continue
            lists = [struct.unpack_from(">3hBBI", keep, p & 0xFFFFFF)[5] for p in rows]
            if all(not dl or dl >> 24 == 0x04 for dl in lists) and sum(1 for dl in lists if dl) == 1:
                found.append(next(dl for dl in lists if dl))
        if len(found) != 1:
            raise BuildError("the arrow's code does not name one skeleton with one list")
        self.tris = _read(link, found[0], "the arrow's list")
        points = [c[1] for _, corners, _ in self.tris for c in corners]
        spans = [max(p[k] for p in points) - min(p[k] for p in points) for k in range(3)]
        feathers = [c[1] for m, corners, both in self.tris if m.cutout and both for c in corners]
        if not (10 <= len(self.tris) <= 80 and spans[0] > 4.0 * max(spans[1:]) and feathers):
            raise BuildError("the arrow's list is not a long thin shape along its X with feathers cut from a texture")
        # the feathers are at the tail; the shaft is what reaches furthest that way with no cut-out; the head is the
        # other end
        low, high = min(p[0] for p in points), max(p[0] for p in points)
        middle = sum(p[0] for p in feathers) / len(feathers)
        solid = [c[1][0] for m, corners, _ in self.tris if not m.cutout for c in corners]
        at_high = middle > (low + high) / 2.0
        self.tail = float(max(solid) if at_high else min(solid))
        self.tip = float(low if at_high else high)
        self.length = abs(self.tail - self.tip)
        if at_high:
            apart = min(p[0] for p in feathers) - low > 0.5 * (high - low)
        else:
            apart = high - max(p[0] for p in feathers) > 0.5 * (high - low)
        if not apart:
            raise BuildError("the arrow's feathers are not all at one end of it")

    def resize(self, times: float) -> None:
        """
        Make it `times` as large every way, about its nock: every corner of its triangles, and
        `tip` and `length` with them. The same in every direction, so its shape and its corners'
        normals are as they were.
        """
        about = self.tail
        self.tris = [
            (
                material,
                [
                    (c[0], (about + (c[1][0] - about) * times, c[1][1] * times, c[1][2] * times), *c[2:])
                    for c in corners
                ],
                both,
            )
            for material, corners, both in self.tris
        ]
        self.tip = about + (self.tip - about) * times
        self.length = abs(self.tail - self.tip)


class Bomb:
    """
    The bomb Link throws, as his game draws it.

    His game does not model the bomb as a ball. Its body is one flat square with a picture of a
    shaded ball on it, turned each frame to face the viewer; only the small cap is a shape. The
    picture is shades of grey with a cut-out edge, and the game adds a colour to it as it draws:
    (flash, 0, 40), where flash goes up and down between nothing and `flash_top` as the fuse runs,
    which is what makes the bomb blink red. So what this hands on is the picture, through that sum,
    for any flash (`picture`); how far from its middle the ball's edge is, in the units Link's own
    limbs are in (`radius`); and the cap's triangles (`cap`).
    """

    def __init__(self, link: Link) -> None:
        self.link = link
        actor = link.rom.actor(BOMB)
        if actor.number != BOMB or actor.held_in != KEEP:
            raise BuildError(f"actor {BOMB:#x} is not drawn from the object every scene shares: not the bomb")
        stores, built = mips.constants(actor.code)
        calls = [w1 for w0, w1 in mips.commands(stores) if w0 == 0xDE000000]
        if len(calls) != 2 or any(c >> 24 != 0x04 for c in calls):
            raise BuildError("the bomb's code does not draw two lists of the shared object")
        self.lists = tuple(calls)
        # The colour: each of the two commands that set one is stored with its second word made of the
        # flash, shifted to the top byte, and a number or-ed in for the two bytes at the bottom (blue,
        # and how solid).
        words = mips.words_of(actor.code)
        added = set()
        for i, _, _, v in stores:
            if v in {0xFA000000, 0xFB000000}:
                near = [
                    words[j] & 0xFFFF
                    for j in range(i, min(i + 14, len(words)))
                    if words[j] >> 26 == 0x0D and words[j] & 0xFFFF
                ]
                if near:
                    added.add(near[0])
        if len(added) != 1:
            raise BuildError("the colour the bomb's code adds to its picture was not found")
        self.base = (0, 0, added.pop() >> 8)
        tops = sorted(mips.to_float(v) for v in built if 0x42800000 <= v <= 0x437F0000 and not v & 0xFFFF)
        if 140.0 not in tops:
            raise BuildError("the bomb's code does not hold 140, the top of its flash")
        self.flash_top = 140
        self._read()

    def _read(self) -> None:
        link = self.link
        first, second = (_read(link, addr, "the bomb's lists") for addr in self.lists)
        flat = [tris for tris in (first, second) if len(tris) == 2]
        if len(flat) != 1:
            raise BuildError("neither of the bomb's lists, or both, is a flat square")
        body = flat[0]
        cap = second if body is first else first
        points = sorted({c[1] for _, corners, _ in body for c in corners})
        material = body[0][0]
        if (
            len(points) != 4
            or len({p[1] for p in points}) != 1
            or not material.texture
            or not material.cutout
            or any(m is not material for m, _, _ in body)
        ):
            raise BuildError("the bomb's body is not a flat square with a cut-out picture")
        half = max(abs(v) for p in points for v in (p[0], p[2]))
        if {(abs(p[0]), abs(p[2])) for p in points} != {(half, half)}:
            raise BuildError("the bomb's body is not a square about its own middle")
        self.half = float(half)
        self._material = material
        # the picture as stored: grey, and solid or not
        w, h, rows = material.stored()
        s0, s1, t0, t1 = material.range
        if (s0, t0) != (0, 0) or not (s1 <= w + 1 and t1 <= h + 1):
            raise BuildError("the bomb's square does not show its picture once")
        self.size = (w, h)
        self._rows = rows
        solid = [(x, y) for y in range(h) for x in range(w) if rows[y][x][3] >= 128]
        if not solid:
            raise BuildError("the bomb's picture is empty")
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        reach = max(math.hypot(x - cx, y - cy) for x, y in solid) + 0.5
        inside = sum(1 for y in range(h) for x in range(w) if math.hypot(x - cx, y - cy) <= reach - 1.5)
        if len(solid) < 0.97 * inside or reach > w / 2.0 + 0.5:
            raise BuildError("the bomb's picture is not a round ball about its middle")
        self.edge = reach / (w / 2.0)  # how far out the ball's edge is, as a part of the square's half
        self.radius = self.half * self.edge
        # the cap: a small shape, lit, at the top of the ball (the body's square lies flat in its list
        # and is stood up by the game, turned a quarter about X, so the square's far edge is the ball's top)
        self.cap = []
        for m, corners, both in cap:
            if not m.lit:
                raise BuildError("the bomb's cap is not a lit shape")
            self.cap.append(
                (m, [(c[1], tuple(v - 256 if v > 127 else v for v in c[4][:3]), c[2]) for c in corners], both),
            )
        ys = [p[1] for _, corners, _ in self.cap for p, _, _ in corners]
        across = max(max(abs(p[0]), abs(p[2])) for _, corners, _ in self.cap for p, _, _ in corners)
        if not (min(ys) > 0.5 * self.radius and across < 0.75 * self.radius and len(self.cap) >= 4):
            raise BuildError("the bomb's second list is not a small cap at the top of the ball")
        self.cap_material = self.cap[0][0]

    def colour(self, flash: int = 0) -> tuple[int, int, int]:
        """The colour the game adds to the picture: (flash, 0, 40)."""
        return (int(flash), *self.base[1:])

    def picture(self, flash: int = 0) -> Image:
        """
        The ball's picture as drawn with `flash` (0 to `flash_top`). Outside the ball the pixels
        are see-through and carry the colour of the nearest one inside.

        Returns:
            (width, height, rows of (r, g, b, a)).
        """
        m = self._material
        kept = m.prim, m.env
        m.prim = m.env = (*self.colour(flash), 255)
        try:
            rows = bleed([[m.colour(px) for px in row] for row in self._rows])
        finally:
            m.prim, m.env = kept
        return self.size[0], self.size[1], rows


class SpinRing:
    """
    The ring of light his spin throws out round him, as his game draws it.

    Found by what his code does, not by a number. The routine that starts a charge (`Link.spins`)
    hands its work on to one that sets the charge to nothing and makes an actor: this one. Its
    drawing routine draws one of two pairs of lists of the object every scene shares, each pair after
    a colour of its own, by a kind it keeps; the kind is set as the spin begins, by whether the
    charge is short of the bigger spin's. So:

    * `lists`, `prim`: the spin's pair, [flat, band], and its colour (r, g, b).
    * `size`: what the drawing scales the lists by, on top of the actor's own size.
    * `flat`, `band`: the spin's two lists as [(three (x, y, z))], in the lists' units: the first
      lying flat, the second a band standing up round its middle. `corners` has them by list, with
      each corner's texture coordinates.

    How it moves, from the routine it runs from the frame after the spin begins:

    * `start`: its size as the spin begins; `grown`: the size it grows to with a grown Link's sword.
    * `ease`, `most`: each frame it grows by that share of the way left, but by no more than `most`.
    * `step`: each frame takes that much off its life, which begins at 1; at nothing it is gone.
    * `solid`, `fade`: while its life is more than `solid` it is drawn wholly; then `fade` times its life.
    * `his_size`: the size every actor of his game is given as it is made, which his code never
      changes: the ring's own units are its lists' times `size` times its size, his are his limbs'
      times this.
    * `made_turned`: how far round from his facing it is made, and `scroll`, `start_offsets`,
      `time_step`: how its textures move round it.
    """

    def __init__(self, link: Link) -> None:
        self.link = link
        rom = link.rom
        found = link.spins()
        words, low = link.code()
        first, end = found["charge_start"]
        field = found["charge_field"]
        # 1. the actor his charge makes: in the routine (of those the start of a charge calls) that sets the charge to
        # nothing
        made = []
        for _, target in mips.calls(words, first, end, low):
            f, e = mips.routine(words, (target - low) // 4)
            zero = {v >> 11 & 31 for v in words[f:e] if v & 0xFFFF07FF == 0x44800000}
            if not any(v >> 26 == 0x39 and v & 0xFFFF == field and v >> 16 & 31 in zero for v in words[f:e]):
                continue
            for i, call in mips.calls(words, f, e, low):
                n = mips.argument(words, i, 6, f)
                if n is not None and 0 < n < 0x400:
                    # (the place is three numbers of his, the first in a3, the other two on the stack)
                    place = [
                        words[j] & 0xFFFF
                        for j in range(max(f, i - 12), i + 2)
                        if words[j] >> 26 == 0x23 and words[j] >> 16 & 31 == 7
                    ]
                    made.append((n, call, place[-1] if place else None))
        if len(set(made)) != 1:
            raise BuildError("the routine that sets Link's charge to nothing does not make one actor")
        number = made[0][0]
        # 2. its code, and its record in the table of actors
        actor = rom.actor(number)
        if actor.number != number or actor.held_in != KEEP:
            raise BuildError(f"actor {number:#x} is not drawn from the object every scene shares")
        code, ram = actor.code, actor.ram
        cw = mips.words_of(code)

        def here(address: int) -> int:
            return (address - ram) // 4

        def fval(i: int, freg: int, floor: int) -> float | None:
            return mips.float_value(cw, i, freg, floor, code, ram)

        # 3. the drawing: a colour (its red, green and blue kept in the code, or-ed with how solid it is), and then two
        # lists before the next colour, is a pair; the first number loaded from its own file is what it scales them by
        df, de = mips.routine(cw, here(actor.draw))
        stores = [s for s in mips.constants(code)[0] if df <= s[0] < de]
        lists = [(i, w1) for i, (w0, w1) in mips.placed_commands(stores) if w0 == 0xDE000000 and w1 >> 24 == 0x04]
        colours = [
            (i, mips.value(cw, i, 1, df))
            for i in range(df, de - 1)
            if cw[i] >> 26 == 0
            and cw[i] & 0x3F == 0x25
            and 1 in {cw[i] >> 21 & 31, cw[i] >> 16 & 31}
            and cw[i + 1] >> 26 == 0x2B
            and cw[i + 1] & 0xFFFF == 4
            and cw[i + 1] >> 16 & 31 == cw[i] >> 11 & 31
        ]
        colours = [(i, c) for i, c in colours if c is not None and c > 0xFFFFFF and not c & 0xFF]
        pairs = {}
        for k, (p, colour) in enumerate(colours):
            after = [w1 for i, w1 in lists if p < i < (colours[k + 1][0] if k + 1 < len(colours) else de)]
            if len(after) == 2:
                pairs[p] = (after, (colour >> 24, colour >> 16 & 255, colour >> 8 & 255))
        loaded = []
        for i in range(df, de):
            if cw[i] >> 26 == 0x31:
                base_at = mips.value(cw, i, cw[i] >> 21 & 31, df)
                if base_at is not None and ram <= (base_at + mips.signed(cw[i])) & 0xFFFFFFFF < ram + len(code) - 3:
                    loaded.append(((base_at + mips.signed(cw[i])) & 0xFFFFFFFF) - ram)
        self.size = struct.unpack(">f", code[loaded[0] : loaded[0] + 4])[0] if loaded else None
        if len(pairs) != 2 or not self.size or not 0.0 < self.size < 1.0:
            raise BuildError(
                f"actor {number:#x} does not draw two pairs of lists, each after a colour, at a size of its own",
            )
        # which pair the kind it keeps picks: the one at the place a test of that byte against 1 goes to
        by_kind, kinds = {}, set()
        for k in range(df, de):
            w = cw[k]
            if w >> 26 != 0x04 or mips.value(cw, k, w >> 16 & 31, k - 3) != 1:
                continue
            reg = w >> 21 & 31
            loads = [j for j in range(k - 1, max(df, k - 8), -1) if mips.writes(cw[j], reg)]
            if not loads or cw[loads[0]] >> 26 != 0x24:
                continue
            target = k + 1 + mips.signed(w)
            hit = [p for p, _ in colours if p >= target][:1]
            if hit and hit[0] in pairs:
                by_kind[1] = hit[0]
                kinds.add(cw[loads[0]] & 0xFFFF)
        if len(by_kind) != 1 or len(kinds) != 1:
            raise BuildError(f"actor {number:#x}'s drawing does not pick one of its pairs by a byte reading 1")
        kind = kinds.pop()
        # 4. that byte is set as the spin begins: 1 when the charge is short of the bigger spin's, nothing when not
        big = struct.unpack("<f", struct.pack("<f", found["big_at"]))[0]
        ways = []
        for i in range(len(cw) - 8):
            w = cw[i]
            if w >> 26 != 0x31 or fval(i + 1, w >> 16 & 31, max(0, i - 4)) != big:
                continue
            fa = w >> 16 & 31
            tests = [
                j
                for j in range(i + 1, i + 5)
                if cw[j] >> 21 == 0x230 and cw[j] & 0x3F == 0x3C and (cw[j] >> 16 & 31) == fa
            ]
            branch = [j for j in range(i + 1, i + 8) if cw[j] >> 16 in {0x4500, 0x4502}]
            if tests and branch:
                b = branch[0]
                ways.append((b + 2, b + 1 + mips.signed(cw[b])))

        def kept(f: int, e: int, fld: int) -> list[int]:
            return [j for j in range(f, min(e, len(cw))) if cw[j] >> 26 == 0x28 and cw[j] & 0xFFFF == fld]

        if len(ways) != 1:
            raise BuildError(f"actor {number:#x}'s code does not test the charge against the bigger spin's once")
        short, other = ways[0]
        rf, re_ = mips.routine(cw, short)
        on_short, on_other = kept(short, other, kind), kept(other, other + 16, kind)
        if (
            not on_short
            or not on_other
            or mips.value(cw, on_short[0], cw[on_short[0]] >> 16 & 31, rf) != 1
            or mips.value(cw, on_other[0], cw[on_other[0]] >> 16 & 31, rf) != 0
        ):
            raise BuildError(f"actor {number:#x}'s code does not keep 1 when the charge is short of the bigger spin's")
        self.lists, self.prim = pairs[by_kind[1]]
        # 5. its size by sword: on each way, a test of the sword it was handed against one number, and a size kept
        # either side
        sizes, swords = {}, set()
        for name, (f, e) in (("spin", (short, other)), ("big", (other, other + 16))):
            tests = []
            for j in range(f, e):
                w = cw[j]
                if w >> 26 != 0x05:
                    continue
                a, b = w >> 21 & 31, w >> 16 & 31
                for x, y in ((a, b), (b, a)):
                    loads = [k for k in range(j - 1, rf - 1, -1) if mips.writes(cw[k], y)]
                    if mips.value(cw, j, x, rf) is not None and loads and cw[loads[0]] >> 26 == 0x24:
                        tests.append((j, mips.value(cw, j, x, rf), cw[loads[0]] & 0xFFFF))
            if len(tests) != 1:
                raise BuildError(
                    f"actor {number:#x}'s code does not test which sword it was handed on the {name}'s way",
                )
            j, one, sword = tests[0]

            def stored(k: int) -> list[int]:
                return [m for m in range(k, min(k + 6, len(cw))) if cw[m] >> 26 == 0x28 and cw[m] & 0xFFFF != kind]

            same, apart = stored(j + 2), stored(j + 1 + mips.signed(cw[j]))
            if not same or not apart or cw[same[0]] & 0xFFFF != cw[apart[0]] & 0xFFFF:
                raise BuildError(f"actor {number:#x}'s code does not keep a size either side of the test of its sword")
            sizes[name] = tuple(mips.value(cw, m, cw[m] >> 16 & 31, rf) for m in (apart[0], same[0]))
            swords.add((one, sword, cw[same[0]] & 0xFFFF))
        if len(swords) != 1:
            raise BuildError(f"actor {number:#x}'s code tests its sword differently on the two ways")
        one, sword, grow_field = swords.pop()
        (self.grown, small), (big_grown, big_small) = sizes["spin"], sizes["big"]
        # the sword it names: the one whose glow it scales least along the blade (each sword's case of the drawing
        # scales by a number)
        glows = {}
        for k in range(df, de):
            w = cw[k]
            if w >> 26 not in {0x04, 0x14}:
                continue
            loads = [j for j in range(k - 1, max(df, k - 8), -1) if mips.writes(cw[j], w >> 21 & 31)]
            if not loads or cw[loads[0]] >> 26 != 0x24 or cw[loads[0]] & 0xFFFF != sword:
                continue
            target = k + 1 + mips.signed(w)
            called = [c for c, _ in mips.calls(cw, target, de, ram)][:2]
            if len(called) == 2:
                x = fval(called[1], 12, called[0])
                if x is not None:
                    glows[mips.value(cw, k, w >> 16 & 31, k - 3)] = abs(x)
        if len(glows) < 3 or min(glows, key=glows.get) != one:
            raise BuildError(
                f"the sword actor {number:#x}'s code sizes alone is not the one it draws the smallest glow for",
            )
        if not (small < self.grown < big_grown and small < big_small):
            raise BuildError(
                f"actor {number:#x}'s sizes are not smaller with the young one's sword and larger with the bigger spin",
            )
        # 6. how it moves: the one routine that reads the size kept for it, whose address the routine that keeps the
        # size builds. It takes a number off a life of its own (a call handed a field of it, nothing and a number),
        # grows its size, at 0x50, towards the size kept (a call handed 0x50 of it and two numbers), then sets its size
        # from that (a call handed what is at 0x50)
        readers = sorted(
            {mips.routine(cw, j) for j in range(len(cw)) if cw[j] >> 26 == 0x24 and cw[j] & 0xFFFF == grow_field},
        )
        if len(readers) != 1 or not any(
            mips.value(cw, j + 1, 5, rf) == ram + 4 * readers[0][0] for j in range(rf, re_)
        ):
            raise BuildError(f"actor {number:#x} has no one routine that grows it to the size kept for it")
        sf, se = readers[0]

        def last_write(i: int, reg: int) -> int | None:
            """The instruction that last sets `reg` for a call at `i` (the one after a call runs first)."""
            if mips.writes(cw[i + 1], reg):
                return cw[i + 1]
            return next((cw[k] for k in range(i - 1, sf - 1, -1) if mips.writes(cw[k], reg)), None)

        life_call = grow_call = size_call = set_size = life = None
        for i, target in mips.calls(cw, sf, se, ram):
            if life_call is None and mips.argument(cw, i, 6, sf) is not None and mips.argument(cw, i, 5, sf) == 0:
                life_call, life = i, last_write(i, 4)  # (addiu a0, s0, life)
            elif (
                grow_call is None
                and last_write(i, 4) == 0x26040050
                and None not in (mips.argument(cw, i, 6, sf), mips.argument(cw, i, 7, sf))
            ):
                grow_call = i  # (addiu a0, s0, 0x50)
            elif grow_call is not None and size_call is None and last_write(i, 5) == 0x8E050050:
                size_call, set_size = i, target  # (lw a1, 0x50(s0))
        if None in (life_call, grow_call, size_call) or life is None or life >> 16 != 0x2604:
            raise BuildError(
                f"actor {number:#x}'s spin does not take a number off a life of its own, grow and take the size",
            )
        self.step = mips.to_float(mips.argument(cw, life_call, 6, sf))
        self.ease, self.most = (
            mips.to_float(mips.argument(cw, grow_call, 6, sf)),
            mips.to_float(mips.argument(cw, grow_call, 7, sf)),
        )

        def own_number(j: int) -> float | None:
            """
            The number a load from the file's own numbers reads: the upper half put in its register
            nearest before it that, with the load's lower half, falls inside the file (a branch
            between may have put another there).

            Returns:
                The number, or None if none is found.
            """
            for k in range(j - 1, max(sf, j - 12) - 1, -1):
                if cw[k] >> 26 == 0x0F and cw[k] >> 16 & 31 == cw[j] >> 21 & 31:
                    at = (((cw[k] & 0xFFFF) << 16) + mips.signed(cw[j])) & 0xFFFFFFFF
                    if ram <= at < ram + len(code) - 3:
                        return struct.unpack(">f", code[at - ram : at - ram + 4])[0]
            return None

        after = [(own_number(j), j) for j in range(size_call, se) if cw[j] >> 26 == 0x31]
        after = [(v, j) for v, j in after if v is not None]
        if len(after) < 2 or abs(after[0][0] * after[1][0] - 1.0) > 1e-4:
            raise BuildError(f"actor {number:#x}'s spin is not drawn wholly and then fading as its life runs out")
        self.solid, self.fade = after[0][0], after[1][0]
        # (where it keeps how much of it is drawn: what the routine stores once it has read those two)
        drawn_field = [cw[j] & 0xFFFF for j in range(after[0][1], se) if cw[j] >> 26 == 0x39 and cw[j] >> 21 & 31 == 16]
        # its size as it is made, and the size every actor is made with: the routine that sets an actor's size stores
        # what it is handed in the three numbers of it at 0x50; the one that runs an actor's own setting up gives every
        # one a size first
        base = link.program_base
        pw = mips.words_of(rom.program)
        s = (set_size - base) // 4
        body = pw[s : s + 7]
        if not (
            body[0] & 0xFFFF07FF == 0x44850000
            and sorted(v & 0xFFFF for v in body if v >> 26 == 0x39 and v >> 21 & 31 == 4) == [0x50, 0x54, 0x58]
        ):
            raise BuildError(
                f"the routine actor {number:#x} sets its size by does not store one number in the three at 0x50",
            )
        start = here(actor.init)
        given = [
            mips.argument(cw, i, 5, start)
            for i, target in mips.calls(cw, start, mips.routine(cw, start)[1], ram)
            if target == set_size
        ]
        if len(given) != 1 or given[0] is None:
            raise BuildError(f"actor {number:#x} is not given a size of its own as it is made")
        self.start = mips.to_float(given[0])
        every = []
        for f, e in mips.routines(pw):
            for k in range(f, e - 3):
                if pw[k] >> 26 != 0 or pw[k] & 0x3F != 0x09:
                    continue  # (jalr: a call through a register)
                loads = [m for m in range(k - 1, max(f, k - 6), -1) if mips.writes(pw[m], pw[k] >> 21 & 31)]
                if not loads or pw[loads[0]] >> 26 != 0x23:
                    continue
                kept_at = pw[loads[0]] & 0xFFFF, pw[loads[0]] >> 21 & 31
                if not any(
                    pw[m] >> 26 == 0x2B and pw[m] >> 16 & 31 == 0 and (pw[m] & 0xFFFF, pw[m] >> 21 & 31) == kept_at
                    for m in range(k + 1, k + 4)
                ):
                    continue  # (not a routine kept in the actor, run and then forgotten)
                every += [
                    mips.to_float(mips.argument(pw, i, 5, f))
                    for i, target in mips.calls(pw, f, k, base)
                    if target == set_size and mips.argument(pw, i, 5, f) is not None
                ]
        mine = [target for _, target in mips.calls(words, 0, len(words), low) if target == set_size]
        if len(every) != 1 or mine:
            raise BuildError("the size every actor is made with was not read, or Link's code sets his own")
        self.his_size = every[0]
        self._read_scroll(cw, code, ram, df, de, drawn_field, here(actor.init))
        # 8. the lists
        self.flat = self.band = None
        self.corners, self._drawn_by = {}, {}
        for list_address in self.lists:
            tris = self._list(list_address)
            heights = sorted({round(p[1], 3) for tri in tris for p in tri})
            if len(heights) == 1 and self.flat is None:
                self.flat = tris
            elif len(heights) == 2 and abs(heights[0] + heights[1]) < 1e-6 and self.band is None:
                self.band = tris
            else:
                raise BuildError(
                    f"the list at {list_address:#x} is neither flat nor a band standing up round its middle",
                )
        if self.flat is None or self.band is None:
            raise BuildError("the spin's two lists are not one flat and one a band")
        # one round of the ring is its first tile's width, once
        widths = {round(self._drawn_by[a][3][1] - self._drawn_by[a][3][0], 6) for a in self.lists}
        tile_widths = {self._drawn_by[a][0][0][1] for a in self.lists}
        if len(widths) != 1 or widths != {float(w) for w in tile_widths}:
            raise BuildError("the ring's lists do not go round once over their first tile's width")
        self.round_texels = widths.pop()

    def _read_scroll(
        self,
        cw: tuple[int, ...],
        code: bytes,
        ram: int,
        df: int,
        de: int,
        drawn_field: list[int],
        init: int,
    ) -> None:
        """
        Read how its textures move round it, and the turn it is made with.

        (a) A time it keeps: a field to which, each frame, what it keeps of how much of it is drawn
        is added twice over. (b) The drawing hands each tile a place to read from, 0xFF less the
        time times a number (its lowest byte), in quarters of a texel. (c) It is made turned about
        the upright from his own facing by a number of its own.

        Raises:
            BuildError: If the code does not do that.
        """

        def fval(i: int, freg: int, floor: int) -> float | None:
            return mips.float_value(cw, i, freg, floor, code, ram)

        def fwrites(w: int, reg: int) -> bool:
            return (
                (w >> 26 == 0x31 and (w >> 16 & 31) == reg)
                or (w >> 26 == 0x11 and w >> 21 & 31 == 0x04 and (w >> 11 & 31) == reg)
                or (w >> 26 == 0x11 and w >> 21 & 31 in {0x10, 0x11, 0x14} and (w >> 6 & 31) == reg)
            )

        def last(j: int, reg: int, floor: int, *, fp: bool = False) -> int | None:
            return next(
                (k for k in range(j - 1, floor - 1, -1) if (fwrites(cw[k], reg) if fp else mips.writes(cw[k], reg))),
                None,
            )

        # (a) `add.s fd, fa, fa` of what is drawn of it, added to the field's own number and stored back in it
        timed = []
        for j in range(8, len(cw) - 6):
            w = cw[j]
            if w >> 21 != 0x230 or w & 0x3F != 0 or (w >> 11 & 31) != (w >> 16 & 31) or len(drawn_field) < 1:
                continue
            src = last(j, w >> 11 & 31, j - 8, fp=True)
            if src is None or cw[src] >> 26 != 0x31 or cw[src] & 0xFFFF != drawn_field[0]:
                continue
            for k in range(j + 1, j + 5):
                v = cw[k]
                if v >> 21 == 0x230 and v & 0x3F == 0 and (w >> 6 & 31) in {v >> 11 & 31, v >> 16 & 31}:
                    other = (v >> 11 & 31) if (v >> 16 & 31) == (w >> 6 & 31) else (v >> 16 & 31)
                    load = last(k, other, j - 8, fp=True)
                    field = cw[load] & 0xFFFF if load is not None and cw[load] >> 26 == 0x31 else None
                    if field is not None and any(
                        cw[m] >> 26 == 0x39 and cw[m] & 0xFFFF == field and (cw[m] >> 16 & 31) == (v >> 6 & 31)
                        for m in range(k + 1, k + 4)
                    ):
                        timed.append(field)
        if len(set(timed)) != 1:
            raise BuildError(
                "the spin's ring keeps no one time that what is drawn of it is added to, twice over, each frame",
            )
        time_field, self.time_step = timed[0], 2.0
        # (b) the call that sets the two tiles' stretches, the first after the time is read: the first tile's place in
        # a2, the second's on the stack at 0x1c
        reads = [j for j in range(df, de) if cw[j] >> 26 == 0x31 and cw[j] & 0xFFFF == time_field]
        call = next((i for i, _ in mips.calls(cw, reads[0], de, ram)), None) if reads else None

        def scrolled(j: int, reg: int) -> tuple[int, float] | None:
            """(n, m) for a register holding n - ((time * m) & 0xFF) at `j`, or None."""
            k = last(j, reg, df)
            w = cw[k] if k is not None else 0
            if w >> 26 != 0 or w & 0x3F != 0x23:  # subu
                return None
            base_value = mips.value(cw, k, w >> 21 & 31, df)
            k2 = last(k, w >> 16 & 31, df)
            if k2 is None or cw[k2] >> 26 != 0x0C or cw[k2] & 0xFFFF != 0xFF:  # andi r, r, 0xff
                return None
            k3 = last(k2, cw[k2] >> 21 & 31, df)
            if k3 is None or cw[k3] >> 21 != 0x220:  # mfc1 r, f
                return None
            k4 = last(k3, cw[k3] >> 11 & 31, df, fp=True)
            if k4 is None or cw[k4] >> 21 != 0x230 or cw[k4] & 0x3F != 0x0D:  # trunc.w.s f, g
                return None
            k5 = last(k4, cw[k4] >> 11 & 31, df, fp=True)
            if k5 is None or cw[k5] >> 21 != 0x230 or cw[k5] & 0x3F != 0x02:  # mul.s g, time, number
                return None
            ops = [cw[k5] >> 11 & 31, cw[k5] >> 16 & 31]
            numbers = [fval(k5, r, df) for r in ops]
            timers = [last(k5, r, df, fp=True) for r in ops]
            kept_time = [t is not None and cw[t] >> 26 == 0x31 and cw[t] & 0xFFFF == time_field for t in timers]
            if sorted(kept_time) != [False, True] or numbers[kept_time.index(False)] is None:
                return None
            return base_value, numbers[kept_time.index(False)]

        stacked = (
            [
                j
                for j in range(reads[0], call)
                if cw[j] >> 26 == 0x2B and cw[j] >> 21 & 31 == 29 and cw[j] & 0xFFFF == 0x1C
            ]
            if call
            else []
        )
        first = scrolled(call + 2 if mips.writes(cw[call + 1], 6) else call, 6) if call else None
        second = scrolled(stacked[-1], cw[stacked[-1]] >> 16 & 31) if stacked else None
        if first is None or second is None:
            raise BuildError("the spin's ring does not move its two tiles by its time")
        self.scroll = (first[1], second[1])
        self.start_offsets = (-first[0] / 4.0, -second[0] / 4.0)
        # (c) the turn about the upright it is made with: his own (+0xb6 of him, which it copies) and so much more
        f0, e0 = mips.routine(cw, init)
        turned = []
        for j in range(f0, e0):
            w = cw[j]
            if w >> 26 == 0x29 and w & 0xFFFF == 0xB6 and w >> 21 & 31 == 16:  # sh r, 0xb6(s0)
                k = last(j, w >> 16 & 31, f0)
                v = cw[k] if k is not None else 0
                if v >> 26 == 0 and v & 0x3F == 0x21:  # addu r, his, n
                    for a, b in ((v >> 21 & 31, v >> 16 & 31), (v >> 16 & 31, v >> 21 & 31)):
                        his, more = last(k, a, f0), mips.value(cw, k, b, f0)
                        if his is not None and cw[his] >> 26 == 0x21 and cw[his] & 0xFFFF == 0xB6 and more is not None:
                            turned.append(more & 0xFFFF)
        if len(set(turned)) != 1:
            raise BuildError("the spin's ring is not made turned from his facing by a number of its own")
        self.made_turned = turned[0] * math.pi / 0x8000

    def _list(self, list_address: int) -> list[tuple[Vector, Vector, Vector]]:
        """
        One of the ring's lists: its triangles, and, kept by list, its corners with their texture
        coordinates and what its two tiles are drawn with. A call into another segment is the
        drawing code's own (its textures' scrolling) and is passed over.

        Returns:
            [(three corners)].

        Raises:
            BuildError: If it is not a shape with a colour, a combiner and textures of the shared object.
        """
        keep = self.link.keep
        verts, found, tiles, tmem = [None] * 64, [], {}, {}
        state = {"env": None, "combine": None, "img": None}

        def walk(off: int) -> None:
            while True:
                w0, w1 = struct.unpack_from(">II", keep, off)
                op = w0 >> 24
                if op == 0x01:
                    n, last = w0 >> 12 & 0xFF, w0 >> 1 & 0x7F
                    if w1 >> 24 != 0x04:
                        raise BuildError("the ring names vertices outside the shared object")
                    for k in range(n):
                        x, y, z, _, s, t = struct.unpack_from(">3hH2h", keep, (w1 & 0xFFFFFF) + 16 * k)
                        verts[last - n + k] = ((float(x), float(y), float(z)), (s, t))
                elif op in {0x05, 0x06}:
                    found.extend(
                        tuple(verts[(word >> sh & 0xFF) // 2] for sh in (16, 8, 0))
                        for word in ((w0,) if op == 0x05 else (w0, w1))
                    )
                elif op == 0xDE:
                    if w1 >> 24 == 0x04:
                        walk(w1 & 0xFFFFFF)
                    if w0 >> 16 & 0xFF == 1:
                        return
                elif op == 0xDF:
                    return
                elif op == 0xFB:
                    state["env"] = (w1 >> 24, w1 >> 16 & 255, w1 >> 8 & 255, w1 & 255)
                elif op == 0xFC:
                    state["combine"] = (w0, w1)
                elif op == 0xFD:
                    state["img"] = w1
                elif op == 0xF5:
                    t = Tile()
                    t.set_tile(w0, w1)
                    tiles[w1 >> 24 & 7] = t
                elif op == 0xF3:
                    tmem[tiles[w1 >> 24 & 7].tmem] = state["img"]
                elif op == 0xF2:
                    tiles[w1 >> 24 & 7].set_size(w0, w1)
                off += 8

        walk(list_address & 0xFFFFFF)
        if any(None in tri for tri in found) or not found or state["env"] is None or state["combine"] is None:
            raise BuildError(f"the list at {list_address:#x} is not a shape with a colour and a combiner of its own")
        self.corners[list_address] = list(found)
        # how it is drawn: tiles 0 and 1 through the combiner, with the spin's colour, over the stretch its corners
        # cover
        drawn = []
        for n in (0, 1):
            t = tiles[n]
            img = tmem.get(t.tmem)
            if img is None or img >> 24 != 0x04:
                raise BuildError(f"the list at {list_address:#x} does not load its textures from the shared object")
            width, height = t.line * (64 >> (2 + t.siz)), (t.lrt >> 2) + 1
            drawn.append((t, width, height, decode_texture(keep, img & 0xFFFFFF, t.fmt, t.siz, width, height)))
        ss = [v[1][0] for tri in found for v in tri]
        ts = [v[1][1] for tri in found for v in tri]
        self._drawn_by[list_address] = (
            drawn,
            combiner(*state["combine"]),
            state["env"],
            (min(ss) / 32.0, max(ss) / 32.0, min(ts) / 32.0, max(ts) / 32.0),
        )
        return [tuple(v[0] for v in tri) for tri in found]

    def _look(self, list_address: int, s: float, t: float, offsets: tuple[float, float]) -> tuple[float, ...]:
        """
        One point of a list as his game draws it, at its own texture coordinates `s`, `t` (texels of
        its first tile): its two tiles through its combiner, with the spin's colour wholly solid,
        each tile read from `offsets` back (its place to read from, in its own texels). The first
        tile is read between its texels, the second at the one the point falls in.

        Returns:
            (r, g, b, a) as floats, 0 to 255.
        """
        tiles, cycles, env, _ = self._drawn_by[list_address]

        def scaled(x: float, shift: int) -> float:
            return x / float(1 << shift) if 0 < shift <= 10 else x * float(1 << (16 - shift)) if shift > 10 else x

        def at(tile: tuple, x: int, y: int) -> tuple:
            tl, w, h, rows = tile
            return rows[min(address(y, h - 1, tl.cmt, tl.maskt), h - 1)][
                min(address(x, w - 1, tl.cms, tl.masks), w - 1)
            ]

        def texel(n: int, *, filtered: bool) -> tuple:
            tile = tiles[n]
            tl = tile[0]
            x, y = scaled(s, tl.shifts) + offsets[n], scaled(t, tl.shiftt)
            if not filtered:
                return at(tile, math.floor(x), math.floor(y))
            x, y = x - 0.5, y - 0.5
            x0, y0 = math.floor(x), math.floor(y)
            fx, fy = x - x0, y - y0
            a, b, c, d = at(tile, x0, y0), at(tile, x0 + 1, y0), at(tile, x0, y0 + 1), at(tile, x0 + 1, y0 + 1)
            return tuple(
                (a[k] * (1 - fx) + b[k] * fx) * (1 - fy) + (c[k] * (1 - fx) + d[k] * fx) * fy for k in range(4)
            )

        rgb, alpha = combine(cycles, texel(0, filtered=True), texel(1, filtered=False), (*self.prim, 255), env)
        return (*rgb, alpha)

    def picture(self, list_address: int, cols: int, lines: int) -> Rows:
        """
        One of the spin's lists as his game draws it on the frame the spin begins, for a texture:
        rows of (r, g, b, a), `cols` round it (one round: the list's coordinates' stretch round,
        which is its first tile's width once) and `lines` across. Each pixel is the average of the
        points of it that fall on each texel of the second tile, the first tile read between its
        texels, each read from where his game has it as the spin begins. A pixel nothing shows in
        takes the colour of what shows nearest round it, so that filtering between the two does not
        darken the edge.

        Returns:
            The rows.
        """
        tiles, _, _, (s0, s1, t0, t1) = self._drawn_by[list_address]
        fx, fy = (round(_scale(tiles[1][0].shifts)), round(_scale(tiles[1][0].shiftt)))
        sub_x = max(1, round((s1 - s0) * fx / cols))
        sub_y = max(1, round((t1 - t0) * fy / lines))
        rows = []
        for j in range(lines):
            row = []
            for i in range(cols):
                acc = [0.0] * 4
                for b in range(sub_y):
                    for a in range(sub_x):
                        s = s0 + (i + (a + 0.5) / sub_x) * (s1 - s0) / cols
                        t = t0 + (j + (b + 0.5) / sub_y) * (t1 - t0) / lines
                        px = self._look(list_address, s, t, self.start_offsets)
                        for k in range(4):
                            acc[k] += px[k] * (px[3] if k < 3 else 1.0)
                n = float(sub_x * sub_y)
                alpha = acc[3] / n
                row.append((*(acc[k] / acc[3] if acc[3] > 0.0 else 0.0 for k in range(3)), alpha))
            rows.append(row)
        for row in rows:  # (colour into the pixels nothing shows in, from the nearest round that does)
            shown = [i for i, px in enumerate(row) if px[3] > 0.0]
            for i, px in enumerate(row):
                if px[3] <= 0.0 and shown:
                    near = min(shown, key=lambda k, i=i: min(abs(k - i), len(row) - abs(k - i)))
                    row[i] = (*row[near][:3], 0.0)
        return [[tuple(int(min(max(v, 0.0), 255.0) + 0.5) for v in px) for px in row] for row in rows]

    def turned(self, n: float) -> float:
        """
        How far round the ring its first texture has moved by time `n` of his game after the spin
        begins (frames; in proportion between two), in radians the way the ring's corners go round
        as their coordinate across it grows less: each frame of the spin his game adds `time_step`
        times how much of the ring is drawn to the time it keeps, and reads the first tile from that
        time times `scroll[0]` quarters of a texel further along; one round is the tile's width.

        Returns:
            The angle.
        """

        def at(k: float) -> float:
            time = 0.0
            for j in range(1, int(k) + 1):
                time += self.time_step * self.frame(j)[1]
            return time

        k = math.floor(n)
        time = at(k) + (at(k + 1) - at(k)) * (n - k)
        return 2.0 * math.pi * time * self.scroll[0] / 4.0 / self.round_texels

    def frame(self, n: float) -> tuple[float, float]:
        """
        The ring at time `n` of his game after the spin begins. It is made on frame 0, which draws
        nothing yet, and from frame 1 it has run its routine n times, until its life runs out;
        between two frames each is taken in proportion.

        Returns:
            (its size, how much of it is drawn).
        """

        def at(k: int) -> tuple[float, float]:
            if k <= 0:
                return self.start, 0.0
            size, life = self.start, 1.0
            for _ in range(k):
                life = max(life - self.step, 0.0)
                if life <= 0.0:
                    return size, 0.0
                size += min((self.grown - size) * self.ease, self.most)
            return size, (1.0 if life > self.solid else life * self.fade)

        k = math.floor(n)
        a, b = at(k), at(k + 1)
        part = n - k
        return a[0] + (b[0] - a[0]) * part, a[1] + (b[1] - a[1]) * part


TRAIL_KIND = 2  # the kind of effect his code adds for the trail his sword leaves


class SwordTrail:
    """
    The trail his sword leaves as he swings it, as his game makes and draws it.

    Found by what his code does, not by a number. As his code sets him up it adds an effect of kind
    `TRAIL_KIND`, handed one of his fields (where the effect's number is kept) and the address of its
    numbers in his own file:

    * `tip`, `base`: the colour (r, g, b, a) of the blade's tip edge and of its other edge where a
      point is added; `tip_end`, `base_end` as a point's life runs out.
    * `life`: how many frames of his a point lasts. Each frame his game adds one to every point's
      age, which starts at `first_age`, and takes the oldest away once its age is more than `life`;
      a point's colour is the start's moved toward the end's by its age over `life`.
    * `steps`: how many steps his game draws between one point and the next, smoothed.

    Where: the program's routine that feeds the effect hands it a point of the blade at each edge, in
    the left hand's own space: `blade_base`, a point it keeps, and `blade_tip`, whose first number is
    the length his game keeps for what he swings (`lengths`, picked by which he holds; by the public
    documentation's order the Master Sword's is the second).

    When: on each frame of a swing his code has the blade out while the swing's frame is from 0 to the
    last frame of its record in the table of swings, and in after; the swing's animation goes on (a
    frame of it a frame of his game: it is played at 2/3 of the speed of his animation routine, which
    goes 3/2 of a frame a frame); then, as he is drawn, a point is added if the blade is out. So the
    points are where the blade is on the swing's frames 1 to the last of its record and one more, as
    far as the swing goes (`frames`).
    """

    def __init__(self, link: Link) -> None:
        rom = link.rom
        words, low = link.code()
        base = link.program_base
        prog = rom.program
        pw = mips.words_of(prog)

        def in_prog(address: int) -> bool:
            return base <= address < base + len(prog)

        def here(address: int) -> int:
            return (address - base) // 4

        # 1. the effect his code adds as it sets him up: a call handed the kind in a2, one of his fields in a1 (s0 and
        # an offset), and on the stack, where the sixth argument goes, the address of its numbers in his own file
        made = []
        for i, target in mips.calls(words, 0, len(words), low):
            if not in_prog(target) or mips.argument(words, i, 6, max(0, i - 24)) != TRAIL_KIND:
                continue
            f, _ = mips.routine(words, i)
            stored = [j for j in range(max(f, i - 24), i + 2) if words[j] >> 21 == 0x57D and words[j] & 0xFFFF == 0x14]
            if not stored:
                continue
            at = mips.value(words, stored[-1], words[stored[-1]] >> 16 & 31, f)
            field = [
                mips.signed(words[k])
                for k in range(max(f, i - 24), i + 2)
                if words[k] >> 26 == 0x09 and words[k] >> 16 & 31 == 5 and words[k] >> 21 & 31 == 16
            ]
            if at is not None and low <= at < low + 4 * len(words) and field:
                made.append((target, at, field[-1]))
        if len(made) != 1:
            raise BuildError(f"Link's code does not add one effect of kind {TRAIL_KIND} with numbers of its own")
        adder, numbers, index_at = made[0]
        code = rom.file(link.move_table()[0])
        off = numbers - low
        calc, flags = struct.unpack_from(">IH", code, off)
        self.tip, self.base, self.tip_end, self.base_end = (tuple(code[off + k : off + k + 4]) for k in (8, 12, 16, 20))
        self.life, mode = code[off + 0x18], code[off + 0x1A]

        # 2. the routine that adds an effect calls the kind's own setting up through a table: an address put together
        # from an upper half and a load, the entries five words each (its size, then its setting up, ending, frame and
        # drawing)
        f, e = mips.routine(pw, here(adder))
        through = []
        for k in range(f, e):
            if pw[k] & 0xFC1FFFFF != 0x0000F809:  # jalr rs
                continue
            reg = pw[k] >> 21 & 31
            load = [j for j in range(k - 1, max(f, k - 8), -1) if mips.writes(pw[j], reg)]
            if load and pw[load[0]] >> 26 == 0x23 and pw[load[0]] >> 21 & 31 == reg:
                up = [
                    j
                    for j in range(load[0] - 1, max(f, load[0] - 6), -1)
                    if pw[j] >> 26 == 0x0F and pw[j] >> 16 & 31 == reg
                ]
                if up:
                    through.append((((pw[up[0]] & 0xFFFF) << 16) + mips.signed(pw[load[0]])) & 0xFFFFFFFF)
        if len(through) != 1:
            raise BuildError("the routine that adds an effect does not call through one table of kinds")
        table = through[0] - 4

        def entry(kind: int) -> tuple[int, ...]:
            return struct.unpack_from(">5I", prog, table - base + 20 * kind)

        if not all(0 < entry(k)[0] < 0x1000 and all(in_prog(a) for a in entry(k)[1:]) for k in range(TRAIL_KIND + 2)):
            raise BuildError("the table of effect kinds is not one of five words a kind")
        _, r_init, _, r_update, r_draw = entry(TRAIL_KIND)

        # 3. its setting up keeps the numbers' life and way of drawing in bytes of the effect (their flags in a half)
        f, e = mips.routine(pw, here(r_init))
        kept = {}
        for k in range(f, e - 1):
            w = pw[k]
            if w >> 26 in {0x24, 0x25} and w >> 21 & 31 == 5:  # lbu / lhu from the numbers, in a1 ...
                for j in range(k + 1, min(e, k + 4)):
                    v = pw[j]
                    if v >> 26 in {0x28, 0x29} and v >> 16 & 31 == w >> 16 & 31 and v >> 21 & 31 == 4:  # ... into a0
                        kept[w & 0xFFFF] = v & 0xFFFF
                        break
        if not {0x18, 0x1A, 4} <= set(kept):
            raise BuildError("the trail's setting up does not keep its life, its way of drawing and its flags")
        life_at, mode_at = kept[0x18], kept[0x1A]

        # 4. a frame of it: a point's age (the word at 4 into it) has one added, and the oldest is taken away when the
        # life, the byte kept, is less than its age (`slt` with the life first)
        f, e = mips.routine(pw, here(r_update))
        aged = [
            k
            for k in range(f, e - 3)
            if pw[k] >> 26 == 0x23
            and pw[k] & 0xFFFF == 4
            and any(
                pw[j] >> 26 == 0x09 and pw[j] >> 21 & 31 == pw[k] >> 16 & 31 and mips.signed(pw[j]) == 1
                for j in range(k + 1, k + 4)
            )
        ]
        older = 0
        for k in range(f, e):
            w = pw[k]
            if w >> 26 == 0 and w & 0x3F == 0x2A:  # slt

                def last(reg: int, k: int = k) -> int | None:
                    return next((pw[j] for j in range(k - 1, max(f, k - 6), -1) if mips.writes(pw[j], reg)), None)

                a, b = last(w >> 21 & 31), last(w >> 16 & 31)
                if (
                    a is not None
                    and b is not None
                    and a >> 26 == 0x24
                    and a & 0xFFFF == life_at
                    and b >> 26 == 0x23
                    and b & 0xFFFF == 4
                ):
                    older += 1
        if not aged or older != 1:
            raise BuildError(
                "a frame of the trail does not age its points by one and take the oldest away past its life",
            )

        # 5. the drawing picks its smoothed way when the way kept is 2 or more (`slti` against 2: the call after the
        # plain one), and that draws between two points through a routine that divides a count by its number of steps
        f, e = mips.routine(pw, here(r_draw))
        smooth = []
        for k in range(f, e - 2):
            if pw[k] >> 26 == 0x24 and pw[k] & 0xFFFF == mode_at:
                for j in range(k + 1, min(e, k + 6)):
                    if pw[j] >> 26 == 0x0A and pw[j] >> 21 & 31 == pw[k] >> 16 & 31 and mips.signed(pw[j]) == 2:
                        later = mips.calls(pw, j, min(e, j + 12), base)
                        if len(later) >= 2:
                            smooth.append(later[1][1])
        if len(smooth) != 1:
            raise BuildError("the trail's drawing does not pick a smoothed way by a way of drawing of 2 or more")
        f, e = mips.routine(pw, here(smooth[0]))
        steps = []
        for _, target in mips.calls(pw, f, e, base):
            g, h = mips.routine(pw, here(target))
            built = mips.floats(pw, g, h)
            for k in range(g, h):
                if pw[k] >> 21 == 0x230 and pw[k] & 0x3F == 3:  # div.s by a number kept in the code
                    for j in range(k - 1, max(g, k - 30), -1):
                        if j in built and built[j][0] == pw[k] >> 16 & 31:
                            steps.append(built[j][1])
                            break
        if len(steps) != 1:
            raise BuildError("the trail's smoothed drawing does not divide by one number of steps")
        self.steps = steps[0]

        # 6. a point is added in one place of the program: a call handed the effect his field names (got by its number)
        # and two places of his, the tip and the base; it starts a point's age at the number it stores at 4 into it
        feeds = []
        for i, _ in mips.calls(pw, 0, len(pw), base):
            if pw[i + 1] >> 26 == 0x23 and pw[i + 1] >> 16 & 31 == 4 and mips.signed(pw[i + 1]) == index_at:
                after = mips.calls(pw, i + 1, min(len(pw), i + 8), base)
                if after and any(pw[j] == 0x00402025 for j in range(i + 1, after[0][0] + 2)):  # move a0, v0
                    feeds.append(after[0])
        if len(feeds) != 1:
            raise BuildError("the program does not add a point to Link's effect in one place")
        add_call, r_add = feeds[0]

        def field_of(i: int, reg: int) -> int | None:
            """The offset of his a call at `i` is handed in `reg`: added to a register a routine keeps (s0 to s7)."""
            for j in [i + 1, *range(i - 1, i - 8, -1)]:
                if mips.writes(pw[j], reg):
                    return mips.signed(pw[j]) if pw[j] >> 26 == 0x09 and 16 <= pw[j] >> 21 & 31 <= 23 else None
            return None

        tip_at = field_of(add_call, 5)
        if tip_at is None or field_of(add_call, 6) is None:
            raise BuildError("the point added is not handed two places of Link's")
        f, e = mips.routine(pw, here(r_add))
        e = next(
            (k + 2 for k in range(f, e) if pw[k] == 0x03E00008),
            e,
        )  # (to its return: a routine with no room on the stack may follow)
        ages = {
            mips.value(pw, k, pw[k] >> 16 & 31, max(f, k - 12))
            for k in range(f, e)
            if pw[k] >> 26 == 0x2B and pw[k] & 0xFFFF == 4
        }
        ages.discard(None)
        if len(ages) != 1:
            raise BuildError("adding a point does not start its age at one number")
        self.first_age = ages.pop()
        f, e = mips.routine(pw, add_call)

        # 7. the blade's two points. The feeding routine hands the record of his weapon (the field 4 before the tip it
        # adds) the tip it was given and, as the fifth argument, a base it works out first: a kept point turned by the
        # matrix of the limb being drawn (the routine that turns a point: handed the point in a0, where it goes in a1)
        feed_calls = mips.calls(pw, f, e, base)
        given = []
        for i, _ in feed_calls:
            if field_of(i, 6) != tip_at - 4:
                continue
            on_stack = [j for j in range(max(f, i - 12), i) if pw[j] >> 21 == 0x57D and pw[j] & 0xFFFF == 0x10]
            if on_stack:
                reg = pw[on_stack[-1]] >> 16 & 31
                source = [
                    pw[j] for j in range(on_stack[-1] - 1, max(f, on_stack[-1] - 8), -1) if mips.writes(pw[j], reg)
                ]
                if source and source[0] >> 26 == 0x09 and source[0] >> 21 & 31 == 29:
                    given.append((i, source[0] & 0xFFFF))
        if len(given) != 1:
            raise BuildError("the feeding routine does not hand Link's weapon's record one base of its own")
        turned = []
        for i, target in feed_calls:
            if i < given[0][0]:
                a1 = [pw[j] for j in [i + 1, *range(i - 1, max(f, i - 5), -1)] if mips.writes(pw[j], 5)]
                if a1 and a1[0] >> 26 == 0x09 and a1[0] >> 21 & 31 == 29 and a1[0] & 0xFFFF == given[0][1]:
                    turned.append((target, mips.handed(pw, i, 4, f)))
        if len(turned) != 1 or turned[0][1] is None or not in_prog(turned[0][1]):
            raise BuildError("the base the feeding routine works out is not one kept point turned")
        r_turn, at = turned[0]
        self.blade_base = struct.unpack_from(">3f", prog, at - base)
        # The tip: the routine run for each limb as it is drawn calls the feeding routine; before the call for a sword
        # it puts a length, from a table picked by which he holds, into the first number of a kept point, and has a
        # routine turn that point first (into the tips it hands on)
        limb = []
        for i, target in mips.calls(pw, 0, len(pw), base):
            if target != base + 4 * f:
                continue
            g, _ = mips.routine(pw, i)
            before = mips.calls(pw, max(g, i - 6), i, base)
            if not before:
                continue
            for k in range(max(g, i - 40), before[-1][0]):
                if pw[k] >> 26 != 0x39:  # swc1 into a kept point ...
                    continue
                up = mips.value(pw, k, pw[k] >> 21 & 31, max(g, k - 6))
                load = [
                    j
                    for j in range(k - 1, max(g, k - 12), -1)
                    if pw[j] >> 26 == 0x31 and pw[j] >> 16 & 31 == pw[k] >> 16 & 31
                ]
                if up is None or not load:
                    continue
                j = load[0]  # ... of a number loaded from a table (an index added)
                ups = [
                    m
                    for m in range(j - 1, max(g, j - 8), -1)
                    if pw[m] >> 26 == 0x0F and pw[m] >> 16 & 31 == pw[j] >> 21 & 31
                ]
                indexed = ups and any(
                    pw[m] >> 26 == 0 and pw[m] & 0x3F == 0x21 and pw[m] >> 11 & 31 == pw[j] >> 21 & 31
                    for m in range(ups[0] + 1, j)
                )
                if indexed:
                    limb.append(
                        (
                            before[-1][1],
                            (up + mips.signed(pw[k])) & 0xFFFFFFFF,
                            (((pw[ups[0]] & 0xFFFF) << 16) + mips.signed(pw[j])) & 0xFFFFFFFF,
                        ),
                    )
        if len(limb) != 1:
            raise BuildError("no one routine puts a length from a table into a point before feeding Link's effect")
        r_tip, point, lengths = limb[0]
        g, h = mips.routine(pw, here(r_tip))
        first = [mips.handed(pw, i, 4, g) for i, target in mips.calls(pw, g, h, base) if target == r_turn]
        if not first or first[0] != point:
            raise BuildError("the routine that works out the tips does not first turn the point the length is put in")
        self.lengths = struct.unpack_from(">6f", prog, lengths - base)
        self.blade_tip = (self.lengths[1], *struct.unpack_from(">2f", prog, point - base + 4))

        # 8. when points are added: a routine of his is handed a swing's first and last frames (bytes 12 and 13 of its
        # record) and 0, and has the blade out while 0 <= the frame he keeps <= the last (f12 the 0, f14 the last)
        _, table_at, _ = link.swing_table()
        outs = set()
        for i, target in mips.calls(words, 0, len(words), low):
            if mips.argument(words, i, 5, max(0, i - 20)) != 0:
                continue
            read = {
                words[j] & 0xFFFF: words[j] >> 21 & 31
                for j in range(max(0, i - 20), i)
                if words[j] >> 26 == 0x24 and words[j] & 0xFFFF in {12, 13}
            }
            if len(read) == 2 and read[12] == read[13]:
                outs.add(target)
        if len(outs) != 1:
            raise BuildError("no one routine is handed a swing's first and last frames and 0")
        f, e = mips.routine(words, (outs.pop() - low) // 4)
        frame = {words[k] >> 16 & 31 for k in range(f, e) if words[k] >> 26 == 0x31 and words[k] >> 21 & 31 == 4}
        compared = [
            (words[k] >> 11 & 31, words[k] >> 16 & 31) for k in range(f, e) if words[k] & 0xFFE0003F == 0x4600003E
        ]  # c.le.s
        moved = {words[k] >> 11 & 31: words[k] >> 16 & 31 for k in range(f, e) if words[k] >> 21 == 0x224}  # mtc1
        if not (
            moved.get(12) == 5
            and moved.get(14) == 7
            and any(a == 12 and b in frame for a, b in compared)
            and any(a in frame and b == 14 for a, b in compared)
        ):
            raise BuildError("the routine handed a swing's frames does not have the blade out from frame 0 to the last")
        # (the routine that starts a swing: it puts the table's address together; its call handed the record's first
        # word, the swing's animation, in a2, goes to one that plays it by its own first call, handed the speed in a3)
        speeds = set()
        for i in range(len(words) - 1):
            if words[i] >> 26 != 0x0F or words[i] & 0xFFFF != (low + table_at + 0x8000) >> 16 & 0xFFFF:
                continue
            if words[i + 1] >> 26 != 0x09 or words[i + 1] & 0xFFFF != (low + table_at) & 0xFFFF:
                continue
            g, h = mips.routine(words, i)
            for k, target in mips.calls(words, g, h, low):
                a2 = [words[j] for j in [k + 1, *range(k - 1, k - 7, -1)] if mips.writes(words[j], 6)]
                if not a2 or a2[0] >> 26 != 0x23 or a2[0] & 0xFFFF or not low <= target < low + 4 * len(words):
                    continue
                gg, hh = mips.routine(words, (target - low) // 4)
                inner = mips.calls(words, gg, hh, low)
                v = mips.handed(words, inner[0][0], 7, gg) if inner else None
                if v is not None:
                    speeds.add(mips.to_float(v))
        if len(speeds) != 1:
            raise BuildError("the routine that starts a swing does not play it at one speed")
        speed = speeds.pop()
        if calc or not flags or mode < 2 or not 1 < self.life < 16 or self.first_age < 0 or not 1 < self.steps < 64:
            raise BuildError("the trail is not made and drawn the way this reads it")
        if abs(speed * 1.5 - 1.0) > 1e-6:
            raise BuildError("a swing is not played a frame of it a frame of his game")

    def span(self) -> int:
        """How many of his frames after a point is added it is still drawn with something of it to see."""
        return self.life - self.first_age

    def colour(self, age: float, edge: float) -> tuple[float, ...]:
        """
        The trail's colour `age` frames of his after a point is added, `edge` of the way from the
        blade's tip edge (0) to its other edge (1). His game truncates each mix; this does not.

        Returns:
            (r, g, b, a), each from 0 to 255.
        """
        share = (self.first_age + age) / float(self.life)
        tip = [a + (b - a) * share for a, b in zip(self.tip, self.tip_end, strict=True)]
        base = [a + (b - a) * share for a, b in zip(self.base, self.base_end, strict=True)]
        return tuple(t + (b - t) * edge for t, b in zip(tip, base, strict=True))

    @staticmethod
    def frames(swing_frames: int, last: int) -> list[int]:
        """
        The frames of a swing (its animation `swing_frames` long, its record's last frame `last`) on which
        a point is added: from the frame after its first to the one after the last, as far as it goes.

        Returns:
            The frames, in order.
        """
        return list(range(1, min(swing_frames - 1, last + 1) + 1))


def _scale(shift: int) -> float:
    """How a tile's shift scales its coordinates."""
    return 1.0 / (1 << shift) if 0 < shift <= 10 else float(1 << (16 - shift)) if shift > 10 else 1.0


def _read(link: Link, list_address: int, what: str) -> list:
    """A list of the shared object, read for his root, refused if it uses a command the reader does not know."""
    kept = link.unknown
    link.unknown = {}
    try:
        tris = link.read_list(list_address, 0)
        if link.unknown:
            raise BuildError(f"{what} use commands this reader does not know")
    finally:
        link.unknown = kept
    return tris
