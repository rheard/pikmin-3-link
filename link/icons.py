"""
Alph's pictures in the HUD and the menus become Link's.

What the game has
-----------------
No leader's picture on screen is drawn from his model. They are small ready-made textures, kept in
each layout archive's one texture file (`lyt/<name>.szs`, then `<name>.arc`, then
`timg/__Combined.bntx`) and named by the leader's letter. Layouts pick one by name:

* `nh_PlayerAR0` to `R4`: his head, turned to the right, with five expressions. The round button
  for him at the bottom left of the HUD and the portrait in a message window both pick among these.
* `nh_PlayerAR5`: the same head knocked out, in grey: two channels, brightness and coverage.
* `nh_PlayerAL0`: his head turned to the left: the D-pad menu, and lists in several menus.
* `nh_PlayerStandA`: head and shoulders, fading out below: his place among the Pikmin types when
  another leader has him in the party. `nt_PlayerStandA` is all of him, standing.
* `nh_MmpAlph`: the emblem that marks him on the map.
* `nt_PlayerArun_00` to `19` and `nt_AnimAlfRun_00` to `19`: him running, as a loop.
* `nh_AnimPlayerAWait`, `Run2` and `Jumpb1` (the versus menu): him standing, running and jumping.

What is written
---------------
Every layout archive that has any of those, with each one drawn again as Link at the same size and
encoded over the old picture in place: same texture, same format, same bytes of file. The other
leaders' pictures are byte for byte the game's, and so is every layout but one.

* The heads, the bust, the standing figure and the loops are his own model from the ROM (`render`),
  with the sword and the Hylian Shield on his back.
* **His face.** His game swaps one of eight pictures for his eyes and one of four for his mouth, so
  each of the leaders' five expressions, and the grey one for when he is down, is a pair of those
  and a tilt of his head (`FACES`). All six are framed alike, so that his head neither shifts nor
  changes size from one to the next.
* **His cap.** Hanging straight down behind him its tail is as long as his head again, and a head
  with it would be half the size in a 64x64 picture. In the pictures of his head alone the tail is
  lifted to trail behind him, as it does when he moves (`CAP`).
* **His own stand, run and backflip.** The standing figure and the bust are the first frame of his
  stand; the running loops are his run, whose 20 frames are the loops' 20; the versus menu's jump is
  the backflip he does with something targeted, and its landing.
* **The head his own game keeps.** Ocarina of Time has one small picture of his head: the marker of
  where he is on a dungeon's map, in the pause screen, 16x16 and facing left (`Rom.map_head`). It is
  the left-facing head and the map's emblem, enlarged by whole pixels and not smoothed.
* The one layout: in the versus menu the light of Alph's antenna is drawn over his tall picture, a
  glow that the picture's animations move about with his head. Link has no antenna, so that light is
  switched off: its panes in Alph's own layout are marked not to be drawn (`put_out_light`).

An archive is not compressed anew: the stock stream is kept but for the stretches that cover the
changed textures (`yaz0.patch`).
"""

from __future__ import annotations

import logging
import math
import re
import struct

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from link import render
from link.errors import BuildError
from link.formats import bntx, sarc, yaz0
from link.geometry import between
from link.oot.link import Link

if TYPE_CHECKING:
    from collections.abc import Iterator

    from link.geometry import Matrix, Vector
    from link.oot.animation import Animation
    from link.oot.graphics import Image, Material, Rows, Vertex
    from link.oot.link import Pose
    from link.oot.rom import Rom
    from link.render import Bounds
    from link.romfs import Mod, RomFs

logger = logging.getLogger(__name__)

LAYOUTS = "lyt"
TEXTURES = "timg/__Combined.bntx"

# Alph's textures, by name without the format suffix, and what to draw for each. A group in the
# pattern is the number of the picture among those of its kind.
PICTURES = tuple(
    (re.compile(pattern + "$"), kind)
    for pattern, kind in (
        (r"nh_PlayerAR(\d)_00", "face"),
        (r"nh_PlayerAL0_00", "glyph"),
        (r"nh_PlayerStandA_00", "bust"),
        (r"nt_PlayerStandA_00", "stand"),
        (r"nh_MmpAlph_00", "emblem"),
        (r"nt_PlayerArun_(\d\d)", "run quarter"),
        (r"nt_AnimAlfRun_(\d\d)", "run side"),
        (r"nh_AnimPlayerAWait_(\d\d)", "big wait"),
        (r"nh_AnimPlayerARun2_(\d\d)", "big run"),
        (r"nh_AnimPlayerAJumpb1_(\d\d)", "big jump"),
    )
)
# Frames in each loop. An archive may hold only one of them (a still), so this cannot be counted there.
LOOPS = {"run side": 20, "run quarter": 20, "big wait": 20, "big run": 26, "big jump": 31}

# His head for each of the leader's face textures: which eyes, which mouth, and how far the head is
# turned from where his stand has it, in degrees about its own three axes (the first tips it towards
# a shoulder, the second turns it, positive towards the viewer, the third nods it, positive down).
# The leaders' own five are: 0 plain, 1 mouth open in surprise, 2 alert, 3 downcast, 4 a scowl; 5 is
# the grey one shown when he is down. His: 1 eyes wide, mouth open, head back; 2 eyes wide, looking
# round; 3 eyes lowered, head down; 4 eyes narrowed, teeth set; 5 eyes screwed shut, mouth open,
# head lolling.
FACES = {
    0: ("open", "closed", (0.0, 0.0, 0.0)),
    1: ("shock", "open", (0.0, 0.0, -8.0)),
    2: ("shock", "closed", (0.0, 10.0, -3.0)),
    3: ("down", "half", (0.0, 0.0, 10.0)),
    4: ("half", "half", (0.0, 0.0, 5.0)),
    5: ("tight", "open", (12.0, 0.0, 8.0)),
}
FACE_VIEW = (32.0, 12.0)  # the camera for the heads: degrees he is turned to the right, degrees it looks down
CAP = (0.0, 0.0, 35.0)  # his cap in the pictures of his head alone: its tail lifted this far, to trail behind
STAND = ("wait", 0.0)  # the animation and frame he stands in
BUST_VIEW = (10.0, 6.0)
BUST_HEAD = 0.58  # the share of the bust picture's height that his head takes, cap and all
BUST_FADE = 0.30  # the share of its height, at the bottom, over which it fades out
STAND_VIEW = (10.0, 6.0)
RUN_VIEWS = {"side": (-90.0, 5.0), "quarter": (-55.0, 10.0)}
# The versus menu's tall pictures (194x342): the leaders stand low in them with room to jump, 122
# pixels tall with their feet at (100, 326). He is a grown man among them, so he stands taller.
BIG_VIEW = (25.0, 8.0)
BIG_HEIGHT = 150 / 342.0
BIG_FEET = (100 / 194.0, 326 / 342.0)
BIG_WAIT = ("wait", 27.0)  # waiting: his stand, this many frames into it and back (where it has him furthest)
BIG_RUN_LOOP = (5, 20)  # the layouts loop frames 5 to 24 of the 26
# The jump: 31 frames, and the leaders' own leave the ground at frame 2 and land at frame 20
# (`BIG_JUMP_AIR`). His backflip is 15 frames: he pushes off in the first, is leaning back by the
# second, turns over, and is down in a crouch at the thirteenth, so the menu's frames in the air are
# its frames 1 to 12 (`BIG_JUMP_OFF`, `BIG_JUMP_DOWN`). His game carries him up meanwhile, which here
# is a parabola of a share of his height (`BIG_JUMP_RISE`). Then the rest of the backflip, and the
# landing that follows it in his game.
BIG_JUMP_AIR = (2, 20)
BIG_JUMP_OFF = 1.0
BIG_JUMP_DOWN = 12.0
BIG_JUMP_RISE = 0.6
EMBLEM_ZOOM = 3  # the map emblem: the map's head at this many pixels a pixel, over a shadow
EMBLEM_SHADOW = ((-3, 3), (10, 56, 18))  # where the shadow falls, and its colour, a dark green

# The light of the leader's antenna in the versus menu. His tall picture there is a small layout of
# its own, with the picture twice (itself and its reflection) and over the head of each a glow, with
# the light itself under it as a part. The picture's animations, whose names begin with
# `LIGHT_ANIMATIONS`, move the glows about with his head. A layout or a layout's animation is a run
# of sections; a pane is a section of at least 0x54 bytes with its flags 8 bytes in (bit 0: it is
# drawn), its opacity 10 in and its name 12 in.
LIGHT_LAYOUT = "blyt/PaAnimIconPlayerA_00.bflyt"
LIGHT_ANIMATIONS = "anim/PaAnimIconPlayerA_00_"
LIGHT_PANES = (re.compile(r"P_Player[A-E]glow_\d\d$"), re.compile(r"Pa_PlayerLight_\d\d$"))
PANES = (b"pan1", b"pic1", b"txt1", b"wnd1", b"bnd1", b"prt1")
PANE_SIZE, PANE_FLAGS, PANE_ALPHA, PANE_NAME = 0x54, 8, 10, 12
MOVED = "FLPA"  # what an animation does to a pane that it only moves, turns and sizes


def to_linear(value: float) -> float:
    """An sRGB value (0-255) as linear light (0-255): how the game stores its grey pictures."""
    value /= 255.0
    return 255.0 * (value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4)


def grey(rows: Rows) -> Rows:
    """
    A colour picture as the two channels of a grey texture: brightness and coverage.

    The knocked-out faces are stored this way. Measured against the same leaders' colour faces,
    their brightness is the colour's luma taken out of sRGB, which is what a renderer that expects
    linear light wants of a texture it does not convert.

    Returns:
        Rows of (brightness, coverage, 0, 255).
    """
    return [
        [(int(to_linear(0.299 * r + 0.587 * g + 0.114 * b) + 0.5), a, 0, 255) for r, g, b, a in row] for row in rows
    ]


def blank(rows: Rows) -> Rows:
    """Fully transparent pixels far from the figure made white, as the game's own pictures have them."""
    return [[px if px[3] or px[:3] != (0, 0, 0) else (255, 255, 255, 0) for px in row] for row in rows]


def paste(rows: Rows, picture: Rows, x: int, y: int) -> Rows:
    """Lay `picture` on `rows` at (x, y), copying only its opaque pixels."""
    for j, line in enumerate(picture):
        for i, px in enumerate(line):
            if px[3]:
                rows[y + j][x + i] = px
    return rows


def _enclose(boxes: list[Bounds]) -> Bounds:
    """The bounds that hold all of several."""
    return (
        min(box[0] for box in boxes),
        max(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


class Model:
    """
    Link as a `render.Figure` takes him: his triangles, all of them or only those wholly of some
    parts, and each material's picture, worked out once.

    His head's list also draws the stub of neck that joins it to his trunk; `parts` leaves out every
    triangle with a corner on a part not named, so a head alone ends at the jaw. `images` is where
    the pictures are kept, shared by every figure of one `Link`.
    """

    def __init__(self, link: Link, parts: tuple[str, ...] | None, images: dict[Material, Image]) -> None:
        self.link, self.parts, self.images = link, parts, images

    def triangles(self, pose: Pose | None = None) -> list[tuple[Material, str, list[Vertex]]]:
        """Every triangle as `Link.triangles` gives them, less any with a corner on a part not named."""
        return [
            (material, part, [corner[:3] for corner in tri])
            for material, part, tri in self.link.skin(pose)
            if self.parts is None or all(corner[3] in self.parts for corner in tri)
        ]

    def image(self, material: Material) -> Image:
        """A material's picture, as `Link.image` gives it."""
        if material not in self.images:
            self.images[material] = self.link.image(material)
        return self.images[material]


class Artist:
    """Draws every kind of picture, from one ROM, at whatever size a texture is."""

    def __init__(self, rom: Rom, samples: int = 4) -> None:
        """
        Check that the ROM has the animations the pictures are drawn from, and read the map's head.

        Raises:
            BuildError: If the table of his animations does not have the ones drawn.
        """
        self.rom = rom
        self.samples = samples  # points averaged per pixel, each way
        self._links: dict[tuple[str, str], Link] = {}
        self._images: dict[Link, dict[Material, Image]] = {}
        self._animations: dict[str, Animation] = {}
        self._pictures: dict[tuple, Rows] = {}
        self._face_box: Bounds | None = None
        self._loops: dict[tuple[str, int], tuple[list[render.Figure], Bounds]] = {}
        missing = sorted({"wait", "run", "backflip", "backflip_land"} - set(self.link().moves()))
        if missing:
            raise BuildError(f"the table of Link's animations in this ROM has no {', '.join(missing)}")
        self.head = rom.map_head()  # the marker of him on a dungeon's map, facing left

    def link(self, eyes: str = "open", mouth: str = "closed") -> Link:
        """His model with one of his pictures of eyes and one of mouths, the sword and shield on his back."""
        if (eyes, mouth) not in self._links:
            self._links[eyes, mouth] = Link(self.rom, eyes=eyes, mouth=mouth)
        return self._links[eyes, mouth]

    def animation(self, which: str) -> Animation:
        """One of his animations, played once through and held at its ends; his run comes round."""
        if which not in self._animations:
            self._animations[which] = self.link().animation(which)
            self._animations[which].loops = which == "run"
        return self._animations[which]

    def picture(self, kind: str, number: int, width: int, height: int, count: int = 1) -> Rows:
        """
        Draw one texture.

        Args:
            kind: What to draw, from `PICTURES`.
            number: The number in the texture's name.
            width: The texture's width.
            height: The texture's height.
            count: How many pictures there are of its kind (the length of a loop).

        Returns:
            Rows of RGBA.

        Raises:
            BuildError: For a kind nobody knows how to draw.
        """
        key = (kind, number, width, height, count)
        if key not in self._pictures:
            match kind.split():
                case ["face"]:
                    rows = self._face(width, height, number)
                case ["bust"]:
                    rows = self._bust(width, height)
                case ["stand"]:
                    rows = self._stand(width, height)
                case ["run", style]:
                    rows = self._run(width, height, number, count, style)
                case ["big", what]:
                    rows = self._big(width, height, number, count, what)
                case ["glyph"]:
                    rows = self._glyph(width, height)
                case ["emblem"]:
                    rows = self._emblem(width, height)
                case _:
                    raise BuildError(f"no way to draw a {kind!r}")
            self._pictures[key] = blank(rows)
        return self._pictures[key]

    def _figure(self, pose: Pose, parts: tuple[str, ...] | None = None, link: Link | None = None) -> render.Figure:
        link = link or self.link()
        return render.Figure(Model(link, parts, self._images.setdefault(link, {})), pose)

    def _draw(
        self,
        figure: render.Figure,
        camera: Matrix,
        width: int,
        height: int,
        scale: float,
        centre: tuple[float, float],
    ) -> Rows:
        return render.draw(figure, camera, width, height, scale, centre, self.samples)

    # -- his head ----------------------------------------------------------------------------------

    def _head(self, turned: Vector, link: Link | None = None) -> render.Figure:
        """His head and cap alone, as he stands, the head turned and the cap's tail lifted."""
        name, frame = STAND
        anim = self.animation(name)
        angles = anim.rotations(frame)
        for part, by in (("head", turned), ("hat", CAP)):
            angles[part] = tuple(a + math.radians(d) for a, d in zip(angles[part], by, strict=True))
        return self._figure(self.link().pose(angles, anim.root(frame)), ("head", "hat"), link)

    def _face(self, width: int, height: int, number: int) -> Rows:
        if number not in FACES:
            raise BuildError(f"no face is set for texture R{number}")
        eyes, mouth, turned = FACES[number]
        camera = render.view(*FACE_VIEW)
        if self._face_box is None:
            # One frame for all six, so that his head does not shift or change size between them.
            self._face_box = _enclose([self._head(t).bounds(camera) for _, _, t in FACES.values()])
        scale, centre = render.fit(self._face_box, width, height, margin=width * 2 / 64.0)
        return self._draw(self._head(turned, self.link(eyes, mouth)), camera, width, height, scale, centre)

    # -- all of him --------------------------------------------------------------------------------

    def _posed(self, which: str, t: float, lift: float = 0.0, spot: tuple[float, float] | None = None) -> Pose:
        """
        His pose at a frame of an animation. `lift` raises him; `spot` is where his root is kept over
        the ground, whatever the animation does with it.

        Returns:
            Every part's rotation and position, as `Link.pose` gives them.
        """
        anim = self.animation(which)
        root = anim.root(t)
        if spot is not None:
            root = (spot[0], root[1], spot[1])
        return self.link().pose(anim.turns(t), (root[0], root[1] + lift, root[2]))

    def _standing(self) -> Pose:
        return self._posed(*STAND)

    def _bust(self, width: int, height: int) -> Rows:
        pose = self._standing()
        camera = render.view(*BUST_VIEW)
        _, _, bottom, top = self._figure(pose, ("head", "hat")).bounds(camera)
        scale = BUST_HEAD * height / (top - bottom)
        margin = height * 3 / 80.0
        # across, he is centred on his trunk: his cap hangs to one side of his head
        left, right, _, _ = self._figure(pose, ("torso",)).bounds(camera)
        centre = ((left + right) / 2.0, top - (height / 2.0 - margin) / scale)
        rows = self._draw(self._figure(pose), camera, width, height, scale, centre)
        start = height * (1.0 - BUST_FADE)
        for y in range(int(start), height):
            keep = max(0.0, 1.0 - (y + 0.5 - start) / (height - 1.0 - start))
            rows[y] = [(r, g, b, int(a * keep + 0.5)) for r, g, b, a in rows[y]]
        return rows

    def _stand(self, width: int, height: int) -> Rows:
        camera = render.view(*STAND_VIEW)
        figure = self._figure(self._standing())
        scale, centre = render.fit(figure.bounds(camera), width, height, margin=2.0)
        return self._draw(figure, camera, width, height, scale, centre)

    def _run(self, width: int, height: int, number: int, count: int, style: str) -> Rows:
        camera = render.view(*RUN_VIEWS[style])
        key = (style, count)
        if key not in self._loops:
            # One cycle of his run over the loop's frames, all framed alike so that he bobs as he does.
            frames = self.animation("run").frames
            figures = [self._figure(self._posed("run", k * frames / float(count))) for k in range(count)]
            self._loops[key] = (figures, _enclose([figure.bounds(camera) for figure in figures]))
        figures, box = self._loops[key]
        scale, centre = render.fit(box, width, height, margin=2.0)
        return self._draw(figures[number], camera, width, height, scale, centre)

    def _big(self, width: int, height: int, number: int, count: int, what: str) -> Rows:
        camera = render.view(*BIG_VIEW)
        left, right, bottom, top = self._figure(self._standing()).bounds(camera)
        scale = BIG_HEIGHT * height / (top - bottom)
        centre = (
            (left + right) / 2.0 - (BIG_FEET[0] - 0.5) * width / scale,
            bottom + (BIG_FEET[1] - 0.5) * height / scale,
        )
        if what == "wait":
            name, turn = BIG_WAIT
            pose = self._posed(name, turn * (1.0 - math.cos(2.0 * math.pi * number / count)) / 2.0)
        elif what == "run":
            first, length = BIG_RUN_LOOP
            pose = self._posed("run", ((number - first) % length) * self.animation("run").frames / float(length))
        else:
            pose = self._jump(number, count, top - bottom)
        return self._draw(self._figure(pose), camera, width, height, scale, centre)

    def _jump(self, number: int, count: int, tall: float) -> Pose:
        """
        His pose at a frame of the versus menu's jump: his backflip on the spot, and its landing.

        The backflip carries his root backwards and his game carries him further; here he comes down
        where he went up, so his root is kept over one spot throughout.

        Returns:
            Every part's rotation and position, as `Link.pose` gives them.
        """
        up, down = BIG_JUMP_AIR
        flip, land = self.animation("backflip"), self.animation("backflip_land")
        name, frame = STAND
        spot = self.animation(name).root(frame)
        spot = (spot[0], spot[2])
        if number < up:
            # Pushing off: in even steps from his stand (where the picture before this one left
            # him), through the backflip's first frame, to the frame he leaves the ground at.
            p = 2.0 * (number + 1) / (up + 1)
            if p >= 1.0:
                return self._posed("backflip", BIG_JUMP_OFF * (p - 1.0), spot=spot)
            stand, turns = self.animation(name), flip.turns(0.0)
            turns = {part: between(turn, turns[part], p) for part, turn in stand.turns(frame).items()}
            height = stand.root(frame)[1]
            return self.link().pose(turns, (spot[0], height + (flip.root(0.0)[1] - height) * p, spot[1]))
        if number < down:  # in the air: the turn, along a parabola
            s = (number - up) / float(down - up)
            t = BIG_JUMP_OFF + (BIG_JUMP_DOWN - BIG_JUMP_OFF) * s
            return self._posed("backflip", t, lift=4.0 * BIG_JUMP_RISE * tall * s * (1.0 - s), spot=spot)
        # down: the rest of the backflip, then the landing, whose first frame is the backflip's last
        rest = flip.frames - 1 - BIG_JUMP_DOWN
        t = (rest + land.frames - 1) * (number - down) / float(count - 1 - down)
        if t <= rest:
            return self._posed("backflip", BIG_JUMP_DOWN + t, spot=spot)
        return self._posed("backflip_land", t - rest, spot=spot)

    # -- from his game's pause screen --------------------------------------------------------------

    def _glyph(self, width: int, height: int) -> Rows:
        """The map's head as large as whole pixels allow, centred."""
        zoom = min(width // len(self.head[0]), height // len(self.head))
        if not zoom:
            raise BuildError(f"a {width}x{height} texture is too small for the map's head")
        head = render.enlarge(self.head, zoom)
        rows = [[(255, 255, 255, 0)] * width for _ in range(height)]
        return paste(rows, head, (width - len(head[0])) // 2, (height - len(head)) // 2)

    def _emblem(self, width: int, height: int) -> Rows:
        """The map's head over a dark copy of its own outline, the way the leaders' map emblems are."""
        head = render.enlarge(self.head, EMBLEM_ZOOM)
        (dx, dy), colour = EMBLEM_SHADOW
        x, y = (width - len(head[0]) - abs(dx)) // 2 + max(0, -dx), (height - len(head) - abs(dy)) // 2 + max(0, -dy)
        if x < 0 or y < 0:
            raise BuildError(f"a {width}x{height} texture is too small for the map emblem")
        rows = [[(*colour, 0)] * width for _ in range(height)]
        shadow = [[(*colour, px[3]) for px in row] for row in head]
        return paste(paste(rows, shadow, x + dx, y + dy), head, x, y)


class Layout:
    """
    One layout archive, unpacked.

    Attributes:
        raw: The `.szs` file.
        plain: The archive inside it.
        inner: The first and last byte, within `plain`, of the archive in it that holds its layouts
            and its texture file, or None if it has none.
        where: The first and last byte of its texture file within `plain`, or None if it has none.
        textures: The textures of that file, by name.
    """

    def __init__(self, name: str, raw: bytes) -> None:
        self.name, self.raw = name, raw
        self.plain = yaz0.decompress(raw)
        outer = sarc.Sarc(self.plain)
        self.inner = self.where = None
        for _, start, end in outer.entries if outer.is_archive else []:
            if self.plain[start : start + 4] != b"SARC":
                continue
            inner = sarc.Sarc(self.plain[start:end])
            if TEXTURES in inner.sizes():
                lo, hi = inner.span(TEXTURES)
                self.inner, self.where = (start, end), (start + lo, start + hi)
        self.textures = bntx.Bntx(self.plain[self.where[0] : self.where[1]]).textures if self.where else {}


def plan(layout: Layout) -> list[tuple[bntx.Texture, str, int, int]]:
    """
    The leader's textures in a layout archive, in the order they lie in the file.

    Returns:
        (texture, kind, number, how many there are of its kind) for each.

    Raises:
        BuildError: If a loop has more frames than it is taken to have.
    """
    found = []
    for pattern, kind in PICTURES:
        count = LOOPS.get(kind, 1)
        for texture in layout.textures.values():
            match = pattern.match(texture.name.split("^")[0])
            if not match:
                continue
            number = int(match.group(1)) if match.groups() else 0
            if kind in LOOPS and number >= count:
                raise BuildError(f"{layout.name}: {texture.name} is frame {number} of a loop taken to have {count}")
            found.append((texture, kind, number, count))
    return sorted(found, key=lambda entry: entry[0].offsets[0])


def encoded(texture: bntx.Texture, rows: Rows) -> Rows:
    """
    A picture in the channels the texture's format keeps.

    Returns:
        The rows to hand the encoder.

    Raises:
        BuildError: For a format none of the leaders' pictures is known to use.
    """
    if texture.codec == "BC3":
        return rows
    if texture.codec == "BC5" and not texture.signed:
        return grey(rows)
    raise BuildError(f"{texture.name} is {texture.describe()}: only colour (BC3) and grey (BC5) pictures are known")


def redraw(layout: Layout, artist: Artist) -> tuple[bytes, int]:
    """
    Draw over every picture of the leader's in a layout archive, in place.

    Returns:
        The archive's new bytes, and how many textures were drawn over.
    """
    lo, hi = layout.where
    data = bytearray(layout.plain[lo:hi])
    textures = plan(layout)
    for texture, kind, number, count in textures:
        rows = artist.picture(kind, number, texture.width, texture.height, count)
        bntx.replace(data, texture, encoded(texture, rows), search=True)
    return layout.plain[:lo] + bytes(data) + layout.plain[hi:], len(textures)


def _sections(data: bytes, what: str) -> Iterator[tuple[int, bytes, int]]:
    """
    Go through the sections of a layout, or of a layout's animation.

    Yields:
        Where each one starts in the file, its kind (four letters) and its size.

    Raises:
        BuildError: If the file is neither, or a section runs past its end.
    """
    if data[:4] not in {b"FLYT", b"FLAN"} or len(data) < 0x14:
        raise BuildError(f"{what} is not a layout or a layout's animation")
    at = struct.unpack_from("<H", data, 6)[0]
    for _ in range(struct.unpack_from("<H", data, 0x10)[0]):
        size = struct.unpack_from("<I", data, at + 4)[0] if at + 8 <= len(data) else 0
        if size < 8 or at + size > len(data):
            raise BuildError(f"{what}: a section runs past the end of the file")
        yield at, data[at : at + 4], size
        at += size


def light_panes(layout: bytes) -> list[tuple[int, str]]:
    """
    The panes of the leader's antenna light in the layout of his picture in the versus menu.

    Returns:
        Where each one starts in the layout, and its name: the glows, and the lights under them.
    """
    panes = []
    for at, kind, size in _sections(layout, LIGHT_LAYOUT):
        if kind in PANES and size >= PANE_SIZE:
            name = layout[at + PANE_NAME : at + PANE_NAME + 24].split(b"\0")[0].decode("latin-1")
            if any(pattern.match(name) for pattern in LIGHT_PANES):
                panes.append((at, name))
    return panes


def animated(animation: bytes, names: set[str], what: str) -> dict[str, list[str]]:
    """
    What a layout's animation does to some of the layout's panes.

    An animation's `pai1` section lists the panes it animates, each with the kinds of thing done to
    it: `FLPA` moving, turning and sizing it, `FLVI` showing and hiding it, `FLVC` its colours and
    opacity.

    Returns:
        {pane: the kinds of thing done to it}, for those of the named panes it animates.
    """
    done = {}
    for at, kind, _ in _sections(animation, what):
        if kind != b"pai1":
            continue
        count, table = struct.unpack_from("<HI", animation, at + 0xE)
        for k in range(count):
            entry = at + struct.unpack_from("<I", animation, at + table + 4 * k)[0]
            name = animation[entry : entry + 28].split(b"\0")[0].decode("latin-1")
            if name in names:
                tags = [
                    entry + struct.unpack_from("<I", animation, entry + 32 + 4 * j)[0]
                    for j in range(animation[entry + 28])
                ]
                done.setdefault(name, []).extend(animation[tag : tag + 4].decode("latin-1") for tag in tags)
    return done


def put_out_light(layout: Layout, plain: bytes) -> tuple[bytes, int]:
    """
    Switch off the leader's antenna light where a layout archive draws it over his picture.

    Only the versus menu's archive does. Each pane of the light is marked not to be drawn and given
    no opacity: two bytes a pane, in place. The light itself is one part that all five leaders'
    pictures share, and it is not touched; nor is any other leader's layout.

    Returns:
        The archive's bytes, and how many panes were switched off: none if it has no such layout.

    Raises:
        BuildError: If the layout does not have a glow, and a light under it, for each picture, or
            if one of the picture's animations does more to them than move them, which would show
            or fade them again.
    """
    start, end = layout.inner
    inner = sarc.Sarc(plain[start:end])
    if LIGHT_LAYOUT not in inner.sizes():
        return plain, 0
    first, _ = inner.span(LIGHT_LAYOUT)
    files = inner.files()
    panes = light_panes(files[LIGHT_LAYOUT])
    glows = [name for _, name in panes if name.startswith("P_")]
    if not glows or len(panes) != 2 * len(glows):
        raise BuildError(f"{layout.name}: {LIGHT_LAYOUT} does not have a glow and a light under it for each picture")
    names = {name for _, name in panes}
    for other, animation in files.items():
        if other.startswith(LIGHT_ANIMATIONS):
            for pane, kinds in animated(animation, names, other).items():
                if set(kinds) - {MOVED}:
                    raise BuildError(f"{layout.name}: {other} does more to {pane} than move it ({', '.join(kinds)})")
    data = bytearray(plain)
    for at, _ in panes:
        data[start + first + at + PANE_FLAGS] &= 0xFE
        data[start + first + at + PANE_ALPHA] = 0
    return bytes(data), len(panes)


def build(rom: Rom, romfs: RomFs, mod: Mod) -> None:
    """
    Write every layout archive that shows the leader into the mod folder, redrawn.

    Raises:
        BuildError: If no archive has a picture of him.
    """
    artist = Artist(rom)
    done = 0
    for name in romfs.names(LAYOUTS, ".szs"):
        layout = Layout(PurePosixPath(name).stem, romfs.read(name))
        if not layout.where or not plan(layout):
            continue
        plain, drawn = redraw(layout, artist)
        plain, lights = put_out_light(layout, plain)
        mod.write(name, yaz0.patch(layout.raw, plain))
        logger.info("  %-16s %d of its %d textures", layout.name, drawn, len(layout.textures))
        if lights:
            logger.info("  %-16s the light of his antenna switched off over his picture: %d panes", "", lights)
        done += 1
    if not done:
        raise BuildError("no layout archive has a picture of the leader")
