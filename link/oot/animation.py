"""
Link's animations, and ways of playing several of them as one.

His animations are not kept with his model. Their headers (a frame count and where the frames are)
sit in the object every scene has loaded, and the frames in a file of their own, 134 bytes each:
the root's position, three angles for each of his 21 limbs, and a word for his face. Everything
here gives the same three things: `frames`, `root(frame)` and `turns(frame)`, the last a rotation
matrix for every limb by name, which `Link.pose` takes.
"""

from __future__ import annotations

import math
import struct

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from link.errors import BuildError
from link.geometry import Matrix, Vector, between, euler_matrix

if TYPE_CHECKING:
    from link.oot.link import Link

PARTS = (
    "root", "waist", "lower",
    "right_thigh", "right_shin", "right_foot", "left_thigh", "left_shin", "left_foot",
    "upper", "head", "hat", "collar",
    "left_upperarm", "left_forearm", "left_hand", "right_upperarm", "right_forearm", "right_hand",
    "sheath", "torso",
)  # fmt: skip

Turns = dict[str, Matrix]


class Animation:
    """
    One of Link's animations: a position for the root and three angles for each limb, every frame.

    It comes round (frame `frames` is frame 0 again) unless `loops` is set false, when it holds its
    last frame instead.
    """

    FRAME = 2 * (3 + 3 * len(PARTS) + 1)

    def __init__(self, link: Link, header: int) -> None:
        self.header = header
        self.frames, zero, where = struct.unpack_from(">hhI", link.keep, header)
        if self.frames <= 0 or zero or where >> 24 != 0x07:
            raise BuildError(f"no animation header at {header:#x}")
        self._d, self._at = link.animation_data(), where & 0xFFFFFF
        self.loops = True

    def frame(self, n: int) -> tuple[int, ...]:
        """Frame `n` as stored: the root's position, each limb's three angles, and the face."""
        return struct.unpack_from(f">{self.FRAME // 2}h", self._d, self._at + n * self.FRAME)

    def _between(self, frame: float) -> tuple[tuple[int, ...], tuple[int, ...], float]:
        last = self.frames - 1
        frame = frame % self.frames if self.loops else min(max(frame, 0.0), float(last))
        lo = math.floor(frame)
        hi = lo + 1 if lo < last else (0 if self.loops else last)
        return self.frame(lo), self.frame(hi), frame - lo

    def root(self, frame: float) -> Vector:
        """Where the root is at a frame; between two frames, in proportion."""
        a, b, f = self._between(frame)
        return tuple(a[k] + (b[k] - a[k]) * f for k in range(3))

    def rotations(self, frame: float) -> dict[str, Vector]:
        """{part: (rx, ry, rz) in radians} at a frame; between two frames, the short way round."""
        a, b, f = self._between(frame)
        out = {}
        for i, name in enumerate(PARTS):
            angles = []
            for k in range(3):
                x, y = a[3 + i * 3 + k], b[3 + i * 3 + k]
                step = (y - x + 0x8000) % 0x10000 - 0x8000
                angles.append((x + step * f) * (2 * math.pi / 65536.0))
            out[name] = tuple(angles)
        return out

    def turns(self, frame: float) -> Turns:
        """
        Every limb's rotation at a frame.

        Between two frames each limb turns the short way from one to the other, which angle by angle
        it would not where the two frames give the same rotation by different angles (some do, half
        a turn apart).

        Returns:
            {part: its rotation as a matrix}.
        """
        a, b, f = self._between(frame)
        out = {}
        for i, name in enumerate(PARTS):

            def one(v: tuple[int, ...], i: int = i) -> Matrix:
                return euler_matrix(*(v[3 + i * 3 + k] * (2 * math.pi / 65536.0) for k in range(3)))

            out[name] = between(one(a), one(b), f) if f else one(a)
        return out


Value = TypeVar("Value")  # what is worked out for a frame: where the root is, or every limb's rotation
Read = Callable[[Animation, float], Value]  # that, from one animation at one of its frames
Mix = Callable[[Value, Value, float], Value]  # two of those put together, a share of the way from the first


def mixed_root(p: Vector, q: Vector, f: float) -> Vector:
    """A root `f` of the way from `p` to `q`."""
    return tuple(x + (y - x) * f for x, y in zip(p, q, strict=True))


def mixed_turns(p: Turns, q: Turns, f: float) -> Turns:
    """Every limb's rotation `f` of the way from `p` to `q`."""
    return {name: between(p[name], q[name], f) for name in p}


class Mixed:
    """
    What a class gets from laying his animations out on one line of frames: `root` and `turns`
    from one `_mix(frame, what, mix)`, which works out a frame by asking `what(animation, frame)`
    of each animation it uses and putting two together with `mix(first, second, share)`.
    """

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        raise NotImplementedError

    def root(self, frame: float) -> Vector:
        """Where the root is at a frame."""
        return self._mix(frame, lambda a, u: a.root(u), mixed_root)

    def turns(self, frame: float) -> Turns:
        """Every limb's rotation at a frame."""
        return self._mix(frame, lambda a, u: a.turns(u), mixed_turns)


class Sequence(Mixed):
    """
    Animations played one after another, as one that comes round.

    Each plays once through, its last frame held for the length of one frame. Where one follows a
    different one he is eased into it over `ease` frames from the pose the one before ended in,
    which is how his own game starts an animation; the first follows the last the same way, so the
    whole comes round. With `once` it does not come round: it starts at its first piece's first
    frame, as that is, and past its end its last frame is held. `travel` is how far (across,
    forward) the whole would leave him from where he should end; that much is taken out of where
    his root is, evenly over the first piece, so that he ends where he stands.
    """

    def __init__(
        self,
        link: Link,
        pieces: tuple,
        ease: int = 6,
        *,
        once: bool = False,
        travel: tuple[float, float] | None = None,
    ) -> None:
        self.pieces = [link.animation(which) for which in pieces]
        self.names = list(pieces)
        for a in self.pieces:
            a.loops = False
        self.starts = [sum(a.frames for a in self.pieces[:i]) for i in range(len(self.pieces))]
        self.frames = sum(a.frames for a in self.pieces)
        self.ease = ease
        self.once = once
        self.travel = travel
        self.window = None

    def _at(self, frame: float) -> tuple[int, float]:
        frame = min(max(frame, 0.0), self.frames - 1.0) if self.once else frame % self.frames
        i = max(k for k, start in enumerate(self.starts) if start <= frame)
        return i, frame - self.starts[i]

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        i, u = self._at(frame)
        now = what(self.pieces[i], u)
        before = i - 1 if i else len(self.pieces) - 1
        if self.names[before] == self.names[i] or u >= self.ease or (self.once and not i):
            return now
        last = self.pieces[before]
        return mix(what(last, float(last.frames - 1)), now, u / float(self.ease))

    def root(self, frame: float) -> Vector:
        """Where the root is at a frame, less the travel taken out."""
        at = super().root(frame)
        if not self.travel:
            return at
        gone = min(1.0, max(0.0, frame) / (self.pieces[0].frames - 1.0))
        return (at[0] - self.travel[0] * gone, at[1], at[2] - self.travel[1] * gone)


class Keys:
    """
    Frames of his animations put end to end as they are, nothing eased and nothing held.

    A jump and its landing are joined this way: the landing's first frame is the frame after the
    jump's last. It does not come round: before its first frame and past its last, those are held.
    Between two frames each limb turns the short way and the root goes straight.
    """

    def __init__(self, link: Link, pieces: tuple) -> None:
        self.names = list(pieces)
        self.pieces = [link.animation(which) for which in pieces]
        self.frames = sum(a.frames for a in self.pieces)
        self._keys = [
            (a.turns(float(n)), tuple(float(c) for c in a.frame(n)[:3])) for a in self.pieces for n in range(a.frames)
        ]

    def _between(self, frame: float) -> tuple[tuple, tuple, float]:
        frame = min(max(float(frame), 0.0), self.frames - 1.0)
        lo = min(math.floor(frame), max(self.frames - 2, 0))
        return self._keys[lo], self._keys[min(lo + 1, self.frames - 1)], frame - lo

    def root(self, frame: float) -> Vector:
        """Where the root is at a frame."""
        (_, a), (_, b), f = self._between(frame)
        return mixed_root(a, b, f)

    def turns(self, frame: float) -> Turns:
        """Every limb's rotation at a frame."""
        (a, _), (b, _), f = self._between(frame)
        return mixed_turns(a, b, f) if f else a
