"""
How the Nintendo 64 draws a textured triangle, as far as Link needs: textures as they are stored,
how a tile repeats, mirrors or clamps them, and the colour combiner that turns a texel into a colour.

The formats (display lists of the later revision of the graphics commands, the colour combiner)
follow the public documentation of the game and the console.
"""

from __future__ import annotations

import struct

from typing import TYPE_CHECKING

from link.errors import BuildError

if TYPE_CHECKING:
    from link.geometry import Vector

Pixel = tuple[int, int, int, int]
Rows = list[list[Pixel]]
Image = tuple[int, int, Rows]  # width, height, rows of RGBA
Vertex = tuple["Vector", "Vector", tuple[float, float]]  # a corner as drawn: position, normal, (u, v)

# Geometry-mode bits a display list sets.
G_CULL_FRONT, G_CULL_BACK, G_LIGHTING, G_TEXTURE_GEN = 0x200, 0x400, 0x20000, 0x40000

# What each number in a combiner word means, by slot. Anything not listed is zero.
_COLOUR_A = {0: "combined", 1: "texel0", 2: "texel1", 3: "prim", 4: "shade", 5: "env", 6: "one"}
_COLOUR_B = {0: "combined", 1: "texel0", 2: "texel1", 3: "prim", 4: "shade", 5: "env"}
_COLOUR_C = {
    0: "combined", 1: "texel0", 2: "texel1", 3: "prim", 4: "shade", 5: "env", 7: "combined_a",
    8: "texel0_a", 9: "texel1_a", 10: "prim_a", 11: "shade_a", 12: "env_a", 14: "prim_lod",
}  # fmt: skip
_COLOUR_D = {0: "combined", 1: "texel0", 2: "texel1", 3: "prim", 4: "shade", 5: "env", 6: "one"}
_ALPHA_ABD = {0: "combined_a", 1: "texel0_a", 2: "texel1_a", 3: "prim_a", 4: "shade_a", 5: "env_a", 6: "one"}
_ALPHA_C = {1: "texel0_a", 2: "texel1_a", 3: "prim_a", 4: "shade_a", 5: "env_a", 6: "prim_lod"}

Formula = tuple[tuple[str | None, ...], tuple[str | None, ...]]  # (a, b, c, d) for colour, then for alpha


def _expand(v: int, bits: int) -> int:
    return v * 255 // ((1 << bits) - 1)


def _rgba16(v: int) -> Pixel:
    r, g, b = v >> 11 & 31, v >> 6 & 31, v >> 1 & 31
    return (r << 3 | r >> 2, g << 3 | g >> 2, b << 3 | b >> 2, 255 if v & 1 else 0)


def decode_texture(
    data: bytes,
    at: int,
    fmt: int,
    siz: int,
    width: int,
    height: int,
    tlut: tuple[bytes, int] | None = None,
) -> Rows:
    """
    A stored texture as rows of RGBA.

    Args:
        data: The file it is in.
        at: Where it starts.
        fmt: 0 RGBA, 2 palette, 3 grey with alpha, 4 grey.
        siz: 0 for 4 bits a texel, 1 for 8, 2 for 16, 3 for 32.
        width: Its width in texels.
        height: Its height in texels.
        tlut: (data, offset) of a palette of 16-bit colours, for a texture that needs one.

    Returns:
        Rows of RGBA.

    Raises:
        BuildError: If it runs past its file, or is in a format not handled here.
    """
    bits = 4 << siz
    stride = (width * bits + 7) // 8
    if at + stride * height > len(data):
        raise BuildError(f"a texture at {at:#x} runs past the end of its file")
    palette = None
    if fmt == 2:
        if tlut is None:
            raise BuildError(f"a texture at {at:#x} needs a palette and none was loaded")
        pd, po = tlut
        palette = [_rgba16(pd[po + 2 * i] << 8 | pd[po + 2 * i + 1]) for i in range(min(256, (len(pd) - po) // 2))]
    rows = []
    for y in range(height):
        row = []
        p = at + y * stride
        for x in range(width):
            if siz == 0:
                v = data[p + x // 2] >> (0 if x & 1 else 4) & 15
            elif siz == 1:
                v = data[p + x]
            elif siz == 2:
                v = data[p + 2 * x] << 8 | data[p + 2 * x + 1]
            else:
                v = struct.unpack_from(">I", data, p + 4 * x)[0]
            if fmt == 0 and siz == 2:
                px = _rgba16(v)
            elif fmt == 0 and siz == 3:
                px = (v >> 24, v >> 16 & 255, v >> 8 & 255, v & 255)
            elif fmt == 2 and siz in {0, 1}:
                px = palette[v]
            elif fmt == 3 and siz == 0:
                i = _expand(v >> 1, 3)
                px = (i, i, i, 255 if v & 1 else 0)
            elif fmt == 3 and siz == 1:
                i = _expand(v >> 4, 4)
                px = (i, i, i, _expand(v & 15, 4))
            elif fmt == 3 and siz == 2:
                px = (v >> 8, v >> 8, v >> 8, v & 255)
            elif fmt == 4 and siz == 0:
                i = _expand(v, 4)
                px = (i, i, i, i)
            elif fmt == 4 and siz == 1:
                px = (v, v, v, v)
            else:
                raise BuildError(f"texture format {fmt} at {bits} bits is not handled")
            row.append(px)
        rows.append(row)
    return rows


class Tile:
    """How a loaded texture is addressed: its format, how it repeats, and the stretch in use."""

    def __init__(self) -> None:
        self.fmt = self.siz = self.line = self.tmem = self.palette = 0
        self.cms = self.cmt = self.masks = self.maskt = self.shifts = self.shiftt = 0
        self.uls = self.ult = self.lrs = self.lrt = 0

    def set_tile(self, w0: int, w1: int) -> None:
        """Take a G_SETTILE command's fields."""
        self.fmt, self.siz, self.line, self.tmem = w0 >> 21 & 7, w0 >> 19 & 3, w0 >> 9 & 0x1FF, w0 & 0x1FF
        self.palette, self.cmt, self.maskt, self.shiftt = w1 >> 20 & 15, w1 >> 18 & 3, w1 >> 14 & 15, w1 >> 10 & 15
        self.cms, self.masks, self.shifts = w1 >> 8 & 3, w1 >> 4 & 15, w1 & 15

    def set_size(self, w0: int, w1: int) -> None:
        """Take a G_SETTILESIZE command's fields: the stretch in use, in quarter texels."""
        self.uls, self.ult, self.lrs, self.lrt = w0 >> 12 & 0xFFF, w0 & 0xFFF, w1 >> 12 & 0xFFF, w1 & 0xFFF

    def key(self) -> tuple[int, ...]:
        """Everything that decides what the tile reads."""
        return (
            self.fmt, self.siz, self.line, self.palette, self.cms, self.cmt, self.masks, self.maskt,
            self.shifts, self.shiftt, self.uls, self.ult, self.lrs, self.lrt,
        )  # fmt: skip


def address(x: int, clamp_hi: int, cm: int, mask: int) -> int:
    """A whole texel coordinate along one axis -> the stored texel it reads, after mirroring and clamping."""
    if cm & 2 or not mask:
        x = min(max(x, 0), clamp_hi)
    if mask:
        size = 1 << mask
        turn = x >> mask
        x &= size - 1
        if cm & 1 and turn & 1:
            x = size - 1 - x
    return x


def combiner(w0: int, w1: int) -> list[Formula]:
    """
    The two formulas of a combiner setting, one for each cycle.

    Returns:
        (colour, alpha) for each cycle, each (a, b, c, d) for (a - b) * c + d.
    """
    w0 &= 0xFFFFFF
    c1 = (
        _COLOUR_A.get(w0 >> 20 & 15),
        _COLOUR_B.get(w1 >> 28 & 15),
        _COLOUR_C.get(w0 >> 15 & 31),
        _COLOUR_D.get(w1 >> 15 & 7),
    )
    a1 = (
        _ALPHA_ABD.get(w0 >> 12 & 7),
        _ALPHA_ABD.get(w1 >> 12 & 7),
        _ALPHA_C.get(w0 >> 9 & 7),
        _ALPHA_ABD.get(w1 >> 9 & 7),
    )
    c2 = (_COLOUR_A.get(w0 >> 5 & 15), _COLOUR_B.get(w1 >> 24 & 15), _COLOUR_C.get(w0 & 31), _COLOUR_D.get(w1 >> 6 & 7))
    a2 = (_ALPHA_ABD.get(w1 >> 21 & 7), _ALPHA_ABD.get(w1 >> 3 & 7), _ALPHA_C.get(w1 >> 18 & 7), _ALPHA_ABD.get(w1 & 7))
    return [(c1, a1), (c2, a2)]


def combine(
    cycles: list[Formula],
    texel0: tuple[float, ...],
    texel1: tuple[float, ...],
    prim: tuple,
    env: tuple,
) -> tuple[tuple[float, float, float], float]:
    """
    Run texels through a combiner's formulas, lighting left out (shade is white).

    Each cycle's result is clamped to 0 to 255, as the hardware clamps it.

    Returns:
        The colour as three floats, and its alpha.
    """
    one = (255.0,) * 3
    values = {
        "texel0": texel0[:3], "texel1": texel1[:3], "prim": prim[:3], "env": env[:3], "shade": one, "one": one,
        "texel0_a": (texel0[3],) * 3, "texel1_a": (texel1[3],) * 3, "prim_a": (prim[3],) * 3, "env_a": (env[3],) * 3,
        "shade_a": one, "prim_lod": (0.0,) * 3, None: (0.0,) * 3,
    }  # fmt: skip
    alphas = {
        "texel0_a": texel0[3], "texel1_a": texel1[3], "prim_a": prim[3], "env_a": env[3], "shade_a": 255.0,
        "one": 255.0, "prim_lod": 0.0, None: 0.0,
    }  # fmt: skip
    rgb, alpha = (0.0, 0.0, 0.0), 0.0
    for colour, alpha_formula in cycles:
        values["combined"], values["combined_a"], alphas["combined_a"] = rgb, (alpha,) * 3, alpha
        p, q, r, u = (values[k] for k in colour)
        rgb = tuple(min(max((p[k] - q[k]) * r[k] / 255.0 + u[k], 0.0), 255.0) for k in range(3))
        p, q, r, u = (alphas[k] for k in alpha_formula)
        alpha = min(max((p - q) * r / 255.0 + u, 0.0), 255.0)
    return rgb, alpha


class Material:
    """
    What a triangle is drawn with: a texture (or none), how it is addressed, and the formula that
    turns a texel into a colour. Two triangles share a material when all of that is equal.

    `texture` is (the data it is in, where in it, its address) and `tlut` (data, where) of its
    palette. `range` is the stretch of texels the material's triangles reach, [s0, s1, t0, t1],
    which grows as triangles are read.
    """

    def __init__(
        self,
        key: tuple,
        texture: tuple[bytes, int, int] | None,
        tile: Tile,
        tlut: tuple[bytes, int] | None,
        combine: list[Formula],
        prim: Pixel,
        env: Pixel,
        mode: int,
        scale: tuple[float, float],
        *,
        lit: bool,
        texgen: bool,
        two_cycle: bool,
    ) -> None:
        self.key = key
        self.texture, self.tile, self.tlut = texture, tile, tlut
        self.combine, self.prim, self.env = combine, prim, env
        self.lit, self.mode, self.texgen, self.two_cycle, self.scale = lit, mode, texgen, two_cycle, scale
        self.range = None
        self.name = f"tex_{texture[2]:08x}" if texture else "flat"
        self._stored = None

    @property
    def cutout(self) -> bool:
        """
        Whether the texture's alpha decides what shows.

        That is a mode that multiplies coverage by alpha (his collar), or one that blends by it (his
        earrings). Everything else is solid whatever its texels' alpha says.
        """
        return bool(self.texture and self.mode & 0x5000)

    def stored(self) -> Image:
        """The texture as it is stored."""
        if self._stored is None:
            t = self.tile
            per_line = {0: 16, 1: 8, 2: 4, 3: 2}[t.siz]  # texels in one 64-bit word of a row
            width = t.line * per_line if t.line else (t.lrs >> 2) + 1
            if t.siz == 3:
                width = t.line * 4 if t.line else width  # 32-bit rows are counted as if 16-bit
            height = max(((t.lrt >> 2) + 1) if not t.maskt else min((t.lrt >> 2) + 1, 1 << t.maskt), 1)
            data, at, _ = self.texture
            self._stored = (width, height, decode_texture(data, at, t.fmt, t.siz, width, height, self.tlut))
        return self._stored

    def texel(self, s: int, t: int) -> Pixel:
        """The stored texel a whole coordinate reads, after repeating, mirroring and clamping."""
        tile = self.tile
        w, h, rows = self.stored()
        x = address(s, (tile.lrs >> 2) - (tile.uls >> 2), tile.cms, tile.masks)
        y = address(t, (tile.lrt >> 2) - (tile.ult >> 2), tile.cmt, tile.maskt)
        return rows[min(y, h - 1)][min(x, w - 1)]

    def colour(self, texel: Pixel) -> Pixel:
        """
        A texel through the combiner.

        Lighting is left out (shade is white): the game that draws him next lights him its own way.

        Returns:
            (r, g, b, a), each from 0 to 255.
        """
        one = (255.0, 255.0, 255.0)
        values = {
            "texel0": texel[:3], "texel1": texel[:3], "prim": self.prim[:3], "env": self.env[:3], "shade": one,
            "one": one, "texel0_a": (texel[3],) * 3, "texel1_a": (texel[3],) * 3, "prim_a": (self.prim[3],) * 3,
            "env_a": (self.env[3],) * 3, "shade_a": one, "prim_lod": (0.0, 0.0, 0.0), None: (0.0, 0.0, 0.0),
        }  # fmt: skip
        alphas = {
            "texel0_a": texel[3], "texel1_a": texel[3], "prim_a": self.prim[3], "env_a": self.env[3],
            "shade_a": 255.0, "one": 255.0, "prim_lod": 0.0, None: 0.0,
        }  # fmt: skip
        combined, combined_a = (0.0, 0.0, 0.0), 0.0
        for colour, alpha in self.combine if self.two_cycle else self.combine[1:]:
            values["combined"], values["combined_a"], alphas["combined_a"] = combined, (combined_a,) * 3, combined_a
            a, b, c, d = (values[n] for n in colour)
            rgb = tuple((a[i] - b[i]) * c[i] / 255.0 + d[i] for i in range(3))
            a, b, c, d = (alphas[n] for n in alpha)
            combined, combined_a = rgb, (a - b) * c / 255.0 + d
        clamped = tuple(int(min(max(v, 0.0), 255.0) + 0.5) for v in combined)
        return (*clamped, int(min(max(combined_a, 0.0), 255.0) + 0.5))


def bleed(rows: Rows) -> Rows:
    """
    Give every pixel that is more than half see-through the colour of the nearest ones that are
    not, keeping its alpha, so that nothing dark creeps in at the edge when the picture is smoothed
    or compressed.

    Returns:
        The new rows.
    """
    h, w = len(rows), len(rows[0])
    rows = [list(row) for row in rows]
    known = [[px[3] >= 128 for px in row] for row in rows]
    while not all(all(row) for row in known):
        fresh = []
        for y in range(h):
            for x in range(w):
                if known[y][x]:
                    continue
                near = [
                    rows[yy][xx]
                    for yy in (y - 1, y, y + 1)
                    for xx in (x - 1, x, x + 1)
                    if 0 <= yy < h and 0 <= xx < w and known[yy][xx]
                ]
                if near:
                    fresh.append((y, x, (*(sum(p[i] for p in near) // len(near) for i in range(3)), rows[y][x][3])))
        if not fresh:
            break  # nothing shows at all
        for y, x, px in fresh:
            rows[y][x] = px
            known[y][x] = True
    return rows
