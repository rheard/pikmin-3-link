"""
Link's bomb: the Ocarina of Time bomb, as the model of the game's second kind of bomb.

The game has two kinds of bomb: the bomb rock, and a second kind (it calls it SearchBomb) that is a
bomb rock underneath and has files of its own, `object/bomb_S.szs` (the model) and
`object/bomb_S_indirect.szs` (what plays which clip, and how large it is to bump into). Only Bingo
Battle uses the second kind, as its mine. Link's bombs are made that kind (`moves.py`), so that they
can look like his and a bomb rock is not touched. In Bingo Battle the mines look like his too.

His game does not model the bomb as a ball: it draws a flat square with a picture of a shaded ball
on it, turned each frame to face the viewer, and a small cap, the one real shape (`oot.actors.Bomb`).
A model here cannot turn to face the viewer, so the ball is built:

* **The ball** is a sphere of the picture's own radius, in two halves. The picture is laid on the
  upper half as if looked down on from above, so that it is seen whole from where this game's camera
  sits, and on the lower half the same, mirrored.
* **Blinking** is two balls in one place: one with the picture as drawn with no flash, one as drawn
  at the top of the flash. Each hangs from a bone of its own, and the clips show one and shrink the
  other to nothing, in turn, faster as the fuse runs down; over the last frames both swell, as his
  game swells the bomb just before it goes off.
* **The cap** is his game's triangles, on top of both balls, its texture laid on by the way each
  corner faces.

In the files: the two balls in two of the kind's five shapes (the other three are emptied); its
colour texture drawn over, its bump map made flat and its shine map "nothing shines"; four numbers
of its material, which make the kind's metal plain (his game does not light the ball: the picture
is the shading); four clips beside the kind's own four, which move nothing but the two balls' sizes,
and which the kind's four states play; and the ball it bumps into made the bomb's size.
"""

from __future__ import annotations

import logging
import math

from typing import TYPE_CHECKING

from link.errors import BuildError
from link.formats import bfres_edit, bntx, sarc, yaz0
from link.formats.bfres import Bfres
from link.geometry import Vector, cross, dot, sub
from link.model import FLAT_NORMAL, _sample
from link.romfs import repack

if TYPE_CHECKING:
    from link.oot.actors import Bomb
    from link.romfs import Mod, RomFs

logger = logging.getLogger(__name__)

ARCHIVE, MODEL_FILE = "object/bomb_S.szs", "object/bomb_S/model.bfres"
SETTINGS = "object/bomb_S_indirect.szs"
CLIP_LIST, COLLISION = "object/bomb_S/anim_define.txt", "object/bomb_S/coll_tree.txt"
# Which of the kind's shapes holds which ball, and the bone each hangs from. The larger shape has
# room for a finer ball; it is the one seen nearly all the time.
DARK, LIT = "body3__body_m", "body1__body_m"
BONES = {DARK: "body3", LIT: "body1"}
FINENESS = {DARK: (10, 20), LIT: (7, 14)}  # rings from pole to middle, and steps round
# The texture is one picture, 256 square: where each thing is drawn in it, in pixels (x, y, size).
TILES = {"dark": (0, 0, 128), "lit": (128, 0, 128), "cap": (0, 128, 64)}
INSET = 1.0 / 32  # how far inside the ball's edge the texture is read, as a part of the picture's half-width
CAP_REACH = 0.8  # the cap's corners read this much of its tile, about the middle
TINY = 1.0 / 1024  # the size of a ball that is not showing
# The clips: ours, the kind's own it stands for, and its last frame (the kind's own lengths, so that
# the markers the list of clips has for each still fall inside it).
CLIPS = (("ob_ignition", "ignition", 89), ("ob_search", "search", 60), ("ob_start", "start", 9), ("ob_wait", "wait", 1))
# How the lit clip blinks: from this frame on, lit for half of every so many frames.
BLINK = ((0, 16), (40, 8), (70, 4))
SWELL = (1.2, 1.4, 1.6)  # the last frames' sizes: his game grows the bomb by a fifth a frame for its last three
SIZE = 2.0  # how many times its size beside him in his own game the bomb is here: twice, so that it is seen
# How the ball takes light: the colour in full, and no reflection, rim or shine added to it (the
# kind's own material is a metal's, which drew the picture silvery and dimmed the blink).
MATERIAL = {
    "color_diff": (1.0, 1.0, 1.0, 1.0),
    "cube_env_map_scale": (0.0,),
    "rim_color": (0.0, 0.0, 0.0),
    "spec_rate": (0.0,),
}
SHINE = (0, 0, 0, 255)  # the shine map: nothing shines (its fourth number is not shine, and stays as the kind has it)

Corner = tuple[Vector, Vector, tuple[float, float]]  # position, normal, texture coordinate (v down)


def _unit(v: Vector) -> Vector:
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def schedule(last: int) -> list[tuple[bool, float]]:
    """For each frame of the lit clip, 0 to `last`: whether the lit ball is the one showing, and its size."""
    out = []
    for f in range(last + 1):
        period = [p for start, p in BLINK if f >= start][-1]
        left = last - f
        size = SWELL[len(SWELL) - 1 - left] if left < len(SWELL) else 1.0
        out.append(((f % period) >= period // 2, size))
    return out


class Shape:
    """The bomb in the game's units, standing on the ground at the model's middle."""

    def __init__(self, bomb: Bomb, scale: float) -> None:
        self.bomb, self.scale = bomb, scale
        self.radius = bomb.radius * scale
        self.centre = (0.0, self.radius, 0.0)
        self.reach = bomb.edge - INSET

    @staticmethod
    def _tile(which: str, u: float, v: float) -> tuple[float, float]:
        x, y, size = TILES[which]
        return ((x + u * size) / 256.0, (y + v * size) / 256.0)

    def ball(self, shape: str) -> list[tuple[Corner, Corner, Corner]]:
        """The ball's triangles for one of the two shapes, each wound to face out."""
        rings, steps = FINENESS[shape]
        which = "dark" if shape == DARK else "lit"
        r, (cx, cy, cz) = self.radius, self.centre
        tris = []
        for half in (1.0, -1.0):

            def corner(i: int, j: int, half: float = half) -> Corner:
                polar, turn = math.pi / 2 * i / rings, 2 * math.pi * (j % steps) / steps
                n = (math.sin(polar) * math.cos(turn), half * math.cos(polar), math.sin(polar) * math.sin(turn))
                # the picture, looked down on: across is x, and down the picture is z; the lower half is
                # the same, mirrored
                uv = self._tile(which, 0.5 + half * n[0] * self.reach / 2, 0.5 + n[2] * self.reach / 2)
                return ((cx + r * n[0], cy + r * n[1], cz + r * n[2]), n, uv)

            for i in range(rings):
                for j in range(steps):
                    a, b, c, d = corner(i, j), corner(i, j + 1), corner(i + 1, j), corner(i + 1, j + 1)
                    quads = [(a, d, c)] if i == 0 else [(a, b, d), (a, d, c)]
                    tris += [tri if half > 0 else tri[::-1] for tri in quads]
        return [self._outward(tri) for tri in tris]

    def _outward(self, tri: tuple[Corner, Corner, Corner]) -> tuple[Corner, Corner, Corner]:
        """A triangle wound so that it faces away from the ball's middle (anticlockwise seen from outside)."""
        (a, _, _), (b, _, _), (c, _, _) = tri
        n = cross(sub(b, a), sub(c, a))
        mid = tuple((a[k] + b[k] + c[k]) / 3 - self.centre[k] for k in range(3))
        return tri if dot(n, mid) > 0 else (tri[0], tri[2], tri[1])

    def cap(self) -> list[tuple[Corner, ...]]:
        """The cap's triangles; one drawn from both sides comes twice, the second turned over."""
        s, (cx, cy, cz) = self.scale, self.centre
        tris = []
        for _, corners, both in self.bomb.cap:
            tri = []
            for p, n, _ in corners:
                n = _unit(n)
                uv = self._tile("cap", 0.5 + n[0] * CAP_REACH / 2, 0.5 - n[1] * CAP_REACH / 2)
                tri.append(((cx + p[0] * s, cy + p[1] * s, cz + p[2] * s), n, uv))
            tris.append(tuple(tri))
            if both:
                tris.append((tri[0], tri[2], tri[1]))
        return tris


def texture(bomb: Bomb) -> list[list[tuple[int, int, int, int]]]:
    """
    The colour texture: the two pictures, each twice its size and smoothed as his game smooths it;
    the cap's texture; the rest the dark ball's darkest colour.

    Returns:
        256 rows of 256 pixels, as (red, green, blue, alpha).
    """
    dark, lit = bomb.picture(0), bomb.picture(bomb.flash_top)
    ground = (*min((px[:3] for row in dark[2] for px in row), key=sum), 255)
    rows = [[ground] * 256 for _ in range(256)]
    m = bomb.cap_material
    w, h, stored = m.stored()
    cap = (w, h, [[(*m.colour(px)[:3], 255) for px in row] for row in stored])
    for which, img in (("dark", dark), ("lit", lit), ("cap", cap)):
        x0, y0, size = TILES[which]
        for y in range(size):
            for x in range(size):
                rows[y0 + y][x0 + x] = _sample(img, (x + 0.5) / size, 1.0 - (y + 0.5) / size)
    return rows


def weld(tris: list[tuple[Corner, ...]]) -> tuple[list[Corner], list[int]]:
    """Turn triangles of corners into vertices and indices."""
    verts, index, indices = [], {}, []
    for tri in tris:
        for pos, nrm, uv in tri:
            key = (tuple(round(c, 4) for c in pos), tuple(round(c, 3) for c in nrm), tuple(round(c, 5) for c in uv))
            if key not in index:
                index[key] = len(verts)
                verts.append((pos, nrm, uv))
            indices.append(index[key])
    return verts, indices


def write_shape(shape: object, verts: list[Corner], indices: list[int]) -> None:
    """
    Overwrite a shape's vertices, triangles and bounds; the vertices left over keep what the game had.

    Raises:
        BuildError: If the new mesh does not fit the shape's buffers.
    """
    if len(verts) > shape.nvert or len(indices) > shape.nindex:
        raise BuildError(f"{shape.name}: {len(verts)} vertices and {len(indices) // 3} triangles do not fit")
    tangents, bitangents = [], []
    for _, nrm, _ in verts:  # (the bump map is flat, so any frame square to the normal lights the same)
        t = _unit(cross((0.0, 1.0, 0.0), nrm)) if abs(nrm[1]) < 0.99 else (1.0, 0.0, 0.0)
        tangents.append(t)
        bitangents.append(_unit(cross(nrm, t)))
    values = {
        "_p0": [v[0] for v in verts],
        "_n0": [v[1] for v in verts],
        "_t0": tangents,
        "_b0": bitangents,
        "_c0": [(1.0, 1.0, 1.0, 1.0)] * len(verts),
        "_u0": [v[2] for v in verts],
    }
    for name in shape.attrs:
        if name not in values:
            raise BuildError(f"{shape.name}: no rule for vertex attribute {name}")
        shape.write(name, values[name])
    shape.set_indices(indices)
    pts = [v[0] for v in verts]
    lo = [min(p[i] for p in pts) for i in range(3)]
    hi = [max(p[i] for p in pts) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    shape.set_bounds(centre, [(hi[i] - lo[i]) / 2 for i in range(3)], max(math.dist(p, centre) for p in pts))


def tracks(skel: object, last: int, *, lit: bool) -> list[tuple]:
    """The tracks of one of our clips: every bone but the root, each with its size and nothing else."""
    plan = schedule(last) if lit else [(False, 1.0)] * (last + 1)
    out = []
    for bone in skel.bones[1:]:
        name = bone["name"]
        if name == BONES[DARK]:
            sizes = [TINY if showing else size for showing, size in plan]
        elif name == BONES[LIT]:
            sizes = [size if showing else TINY for showing, size in plan]
        else:
            sizes = [TINY] * (last + 1)  # the kind's other parts: their shapes are empty
        if len(set(sizes)) == 1:
            out.append((name, (sizes[0],) * 3, (0.0, 0.0, 0.0), None, {}))
        else:
            out.append((name, (sizes[0],) * 3, (0.0, 0.0, 0.0), None, {4: sizes, 8: sizes, 0xC: sizes}))
    return out


def clip_list(text: bytes) -> bytes:
    """
    The kind's list of clips with its four states playing ours: the second name of each of the
    first four entries (the tag is kept, since the game's code asks for a clip by its tag).

    Returns:
        The changed list.

    Raises:
        BuildError: If the list is not laid out as expected.
    """
    head, opened, rest = text.partition(b"{")
    body, closed, tail = rest.partition(b"\r\n}\r\n")  # the first section, which is the skeletal clips
    if not opened or not closed:
        raise BuildError("the bomb's list of clips is not laid out as expected")
    for ours, theirs, _ in CLIPS:
        old = f'"{theirs}" \r\n\t\t"{theirs}" \r\n'.encode()
        if body.count(old) != 1:
            raise BuildError(f"the bomb's list of clips does not have {theirs} once among the skeletal clips")
        body = body.replace(old, f'"{theirs}" \r\n\t\t"{ours}" \r\n'.encode())
    return head + opened + body + closed + tail


def collision(text: bytes, radius: float) -> bytes:
    """
    The file that says what the kind bumps into, made a ball of `radius` on the ground.

    Returns:
        The changed file.

    Raises:
        BuildError: If the file is not laid out as expected.
    """
    lines = text.split(b"\r\n")
    at = [i for i, line in enumerate(lines) if line.endswith(b"# offset")]
    if len(at) != 1 or not lines[at[0] + 1].endswith(b"# radius"):
        raise BuildError("the bomb's file of what it bumps into is not laid out as expected")
    lines[at[0]] = f"\t0.0 {round(radius, 6)!r} 0.0 # offset".encode()
    lines[at[0] + 1] = f"\t{round(radius, 6)!r} # radius".encode()
    return b"\r\n".join(lines)


def remodel(stock: bytes, bomb: Bomb, scale: float) -> bytes:
    """
    The kind's model file with his bomb in it.

    Returns:
        The new model file.

    Raises:
        BuildError: If the kind's model is not put together the way this expects.
    """
    shape = Shape(bomb, scale)
    f0 = Bfres(bytearray(stock))
    skel = f0.models()[0].skel
    theirs = {a.name: a for a in f0.skel_anims()}
    for _, like, last in CLIPS:
        if like not in theirs or theirs[like].frames != last:
            raise BuildError(f"the bomb's own clip {like} is not {last + 1} frames long")
    if [b["name"] for b in skel.bones] != ["bomb_S", "body1", "body2", "body3", "body4"] or any(
        b["parent"] != 0 for b in skel.bones[1:]
    ):
        raise BuildError("the bomb's bones are not the five expected")
    grower = bfres_edit.Grower(stock)
    new = grower.add_clips([(ours, last, theirs[like].o) for ours, like, last in CLIPS])
    for off, (ours, _, last) in zip(new, CLIPS, strict=True):
        grower.add_tracks(off, tracks(skel, last, lit=ours == "ob_ignition"))
    data = bytearray(grower.finish())
    f = Bfres(data)
    model = f.models()[0]
    shapes = {sh.name: sh for sh in model.shapes}
    textures = bntx.Bntx(data)
    for name in (DARK, LIT):
        sh = shapes[name]
        if skel.bones[sh.bone]["name"] != BONES[name] or sh.skin != 0 or model.mats[sh.mat].name != "body_m":
            raise BuildError(f"the shape {name} does not hang from {BONES[name]} alone, with the body's material")
        tris = shape.ball(name) + shape.cap()
        verts, indices = weld(tris)
        write_shape(sh, verts, indices)
    for sh in model.shapes:
        if sh.name not in {DARK, LIT}:
            sh.set_indices([])
    material = next(m for m in model.mats if m.name == "body_m")
    colour = textures[material.role["_a0"]]
    if (colour.width, colour.height) != (256, 256):
        raise BuildError("the bomb's colour texture is not 256 square")
    bntx.replace(data, colour, texture(bomb), search=True)
    bntx.fill(data, textures[material.role["_n0"]], FLAT_NORMAL)
    if material.role["_r0"] != material.role["_s0"]:
        raise BuildError("the bomb's shine and roughness are not one texture")
    bntx.fill(data, textures[material.role["_s0"]], SHINE)
    for name, values in MATERIAL.items():
        material.set_param(name, values)
    logger.info("  the ball is %.2f units across", 2 * shape.radius)
    return bytes(data)


def build(bomb: Bomb, scale: float, romfs: RomFs, mod: Mod) -> float:
    """
    Write the bomb's two files into the mod folder. `scale` is the game's units to one of his model's.

    Returns:
        The ball's radius as the file of what it bumps into gives it.
    """
    archive, settings = romfs.read(ARCHIVE), romfs.read(SETTINGS)
    stock = sarc.Sarc(yaz0.decompress(archive)).files()[MODEL_FILE]
    mod.write(ARCHIVE, repack(archive, {MODEL_FILE: remodel(stock, bomb, scale * SIZE)}))
    files = sarc.Sarc(yaz0.decompress(settings)).files()
    radius = round(bomb.radius * scale * SIZE, 6)
    changes = {
        CLIP_LIST: clip_list(files[CLIP_LIST]),
        COLLISION: collision(files[COLLISION], bomb.radius * scale * SIZE),
    }
    mod.write(SETTINGS, repack(settings, changes))
    return radius
