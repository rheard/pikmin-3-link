"""
Alph becomes Link: the model, his animations, and the game's list of clips.

What is written
---------------
* `player/pikminAll.szs`: the one model file every leader and Pikmin lives in, with Alph's model
  rebuilt as Link, bones of his own added, and his clips.
* `player/pikminAll_indirect.szs`: the game's list of animation clips, with his clips on the end
  and the leaders' standing entry pointed at his own standing clip.
* `effect/effect_common.szs`: the effects, with the light of Alph's antenna put out.

How
---
* **Skeleton.** The leaders' clips (shared with every Pikmin) turn and scale each bone but move only
  the root, so where a limb joins the body comes from each model's own skeleton. Alph's joints are
  moved to Link's hips, neck and shoulders, their rest rotations kept, and Link is bound in a rest
  pose that matches a leader's: standing, arms straight out.
* **Bones of his own.** A leader's arm or leg is one bone; Link's has an elbow or a knee. Each of his
  parts gets a bone of its own, hung under the leader bone it would otherwise have been welded to,
  and every clip in the file a track that tells it to stay at rest, so he moves exactly as if welded.
  `formats/bfres_edit.py` grows the file for that.
* **His legs keep walking.** While a leader carries a Pikmin the game plays one clip on his upper
  body and the walk on his legs: two layers, each driving a run of bones from a named one to the end
  of its branch. A bone can only be added at the end of the list, so Alph's three leg bones are
  renamed out of use and copies under their names put after his upper body's new bones, with his
  legs' bones after those.
* **Clips of his own.** In the leaders' walk and run his bones carry curves as well: for every frame,
  where his own animation puts that part, in the space of the leader bone above it. Alph's bones
  still do the leaders' walk underneath; his cancel it and replace it. Nobody else has bones by
  those names, so the same clips play on everyone else unchanged.
* **His standing still** is longer than the leaders' clip for it, so it is a clip of his own, and
  the game's list's entry for a leader standing is pointed at it. For the leaders' own bones it holds
  the stock clip, round and round, so the other leaders stand as they always have.
* **Moves.** His swings, roll, bombs, idles, bow and spin are new clips, added to the file and to
  the end of the game's list. `moves.py` is the code that plays them.
* **Looks.** What his scabbard and hands hold changes in some clips. Each thing they can be drawn as
  is on a small bone of its own, under the part's, which a clip shows at its size or shrinks to a
  thousandth.
* **The trail of his sword.** A ribbon of points, each on a bone of its own that only the clips of
  his swings and his spin move, so that it costs every other animation nothing.
* **Mesh and textures.** Alph's body and head shapes are overwritten with Link's triangles, each
  vertex following one bone, and his materials packed into one image per shape over Alph's textures.
  Every other shape is emptied, but the antenna bulb's, which draws the ring of his spin, and the
  eyes', which draws the trail of his sword.
* **Antenna light.** With the antenna gone its light would float over his head, so Alph's five light
  effects are made invisible.
"""

from __future__ import annotations

import logging
import math
import re
import struct

from typing import TYPE_CHECKING, Any

from link.errors import BuildError
from link.formats import bfres_edit, bntx, sarc, vfxb, yaz0
from link.formats.bfres import Bfres, Model, Shape, SkelAnim, Skeleton
from link.geometry import (
    Matrix,
    Vector,
    cross,
    dot,
    euler_matrix,
    matmul,
    matrix_euler,
    sub,
    to_local,
    to_model,
    transpose,
    unit,
)
from link.romfs import repack

if TYPE_CHECKING:
    from link.character import Character, Corner, Move, Stand
    from link.oot.graphics import Image, Material
    from link.romfs import Mod, RomFs

logger = logging.getLogger(__name__)

ARCHIVE = "player/pikminAll.szs"
SETTINGS = "player/pikminAll_indirect.szs"
EFFECTS = "effect/effect_common.szs"
MODEL_FILE = "model.bfres"
CLIP_LIST = "anim_define.txt"
EFFECT_FILE = "effect_common.ptcl"

LEADER = "A"  # Alph, whose model is `playerA`
# The lights the game hangs on the tip of a leader's antenna: the steady glow, and the flares for
# the whistle, dismiss, charge and dodge. Each leader has a copy of each, so one's can be put out.
LIGHTS = (
    "c3_player{}_light",
    "c3_light_whistle_per_player{}",
    "c3_light_kaisan_per_player{}",
    "c3_light_totsugeki_per_player{}",
    "c3_light_sidestep_per_player{}",
)
# The shapes that take the new mesh, by material. Every shape keeps its own material: as it sets a
# model up the game makes a thing for each shape and stores it in that shape's material, and then
# goes through every material using the thing stored there, so a material left with no shape stops
# the game as a map loads.
BODY_MATERIALS = ("body_m",)
HEAD_MATERIALS = ("head_m", "face_m")
RIG = ("sebonjnt", "headjnt", "lhandjnt", "rhandjnt", "llegjnt", "rlegjnt")  # the leader bones his parts hang from
OWN = "m_"  # the prefix of the bones added for his parts
OLD = "x_"  # the prefix a leader's bone is renamed with when a copy further down the list takes its name
# The game animates a leader in two layers when he does one thing with his arms and another with his
# legs. Each layer drives a run of bones: from the bone of this name to the end of its branch.
LEGS_ROOT = "legcentre"
HIDDEN = 0.001  # the size a clip gives the bone of a look that is not to show
# The leaders' punch draws its spark 4 units along the right arm's bone. In a move that bone has
# nothing to carry, so it is put where the spark belongs: on the blade.
HIT_BONE, HIT_REACH = "rhandjnt", 4.0
ACTION, MORE = 1024, 2048  # marker flags in the game's list of clips
LOOP_START, LOOP_END = 2, 4  # ... and the two that make an entry go round: from the second it goes back to the first

# Flat material values for him (0-255). The specular map's alpha masks diffuse light on the stock
# leaders (0 only on the glass of the suit's gauges), so it stays near their usual value.
SPECULAR = (64, 64, 64, 240)
ROUGHNESS = (255, 255, 255, 255)
FLAT_NORMAL = (128, 128, 255, 255)
SOLID_TILE = 32  # pixels a single-colour material gets in the atlas
GUTTER = 2  # pixels between tiles
MARGIN = 2  # source texels baked around the part of a texture the triangles use

# Curve targets in a clip's track: the offset of the value each curve drives.
SCALE_CURVES = (0x4, 0x8, 0xC)
TRANSLATE_CURVES = (0x10, 0x14, 0x18)
ROTATE_CURVES = (0x20, 0x24, 0x28)

Frame = tuple[Matrix, Vector]  # where something sits: its rotation and its origin
Plan = dict[int, tuple[Vector, Vector]]  # bone index -> its translation from its parent, its origin
Pose = dict[str, dict[str, Any]]  # bone -> {"S" | "R" | "T": values}, for the bones a clip moves
Target = tuple[Matrix, Vector, Matrix, Vector]  # the leader bone above a part: rotation, origin; the part's
Track = tuple[str, Vector, Vector, Vector | None, dict[int, list[float]]]  # bone, scale, rotation, place, curves


# -- the skeleton ----------------------------------------------------------------------------------


def plan_skeleton(skel: Skeleton, char: Character) -> tuple[list[Frame], Plan]:
    """
    New rest positions for the leader's bones.

    Rotations are kept, so each bone's rest orientation, and with it what an animation's angles
    mean, is what it was.

    Returns:
        The stock skeleton's rest pose, and the plan.

    Raises:
        BuildError: If the skeleton is not the one the rig was written for.
    """
    world = skel.world()
    names = [b["name"] for b in skel.bones]
    for b in skel.bones:
        if any(abs(s - 1.0) > 1e-6 for s in b["scale"]):
            raise BuildError(f"bone {b['name']} has a rest scale, which is not handled")
    missing = sorted(set(RIG) - set(names))
    if missing:
        raise BuildError(f"the leader has no bone named {', '.join(missing)}")
    target = dict(char.pivot)
    target[LEGS_ROOT] = char.pivot["sebonjnt"]  # the legs' parent turns about the pelvis too
    # The antenna is gone; fold its bones onto the top of the head so anything the game hangs on
    # the tip sits there. A little length is left in case code normalises it.
    x, y, z = char.top
    target["happajnt1"], target["happajnt2"], target["happajnt3"] = (x, y, z), (x, y + 0.1, z), (x, y + 0.2, z)
    plan: Plan = {}
    for i, b in enumerate(skel.bones):
        parent = b["parent"]
        if parent < 0:
            # The root's translation is animated (it is the hip height), so it is left alone.
            plan[i] = (tuple(b["trans"]), tuple(world[i][1]))
            continue
        prot, porigin = world[parent][0], plan[parent][1]
        if b["name"] in target:
            origin = target[b["name"]]
            local = to_local(prot, porigin, origin)
        else:
            local = tuple(b["trans"])
            origin = to_model(prot, porigin, local)
        plan[i] = (local, origin)
    return world, plan


def write_skeleton(f: Bfres, skel: Skeleton, plan: Plan) -> tuple[Skeleton, list[Frame]]:
    """
    Store the leader bones' new translations, then every skinned bone's inverse rest matrix.

    Returns:
        The skeleton read back as stored, and its rest pose.
    """
    for i, (local, _) in plan.items():
        b = skel.bones[i]
        struct.pack_into("<3f", f.d, b["off"] + 0x54, *local)
        # A stock bone at its parent's origin carries a "translation is zero" hint; clear it once it moves.
        flags = b["flags"] & ~0x08000000 if any(abs(c) > 1e-9 for c in local) else b["flags"]
        struct.pack_into("<I", f.d, b["off"] + 0x34, flags)
    skel = Skeleton(f, skel.off)
    world = skel.world()
    for k, bone in enumerate(skel.matrix_bones[: skel.nsmooth]):
        rot, origin = world[bone]
        rows = []
        for r in range(3):
            rows += [
                rot[0][r],
                rot[1][r],
                rot[2][r],
                -(rot[0][r] * origin[0] + rot[1][r] * origin[1] + rot[2][r] * origin[2]),
            ]
        struct.pack_into("<12f", f.d, skel.inverse_off + k * 48, *rows)
    return skel, world


def branch_end(bones: list[dict], i: int) -> int:
    """
    Where the game ends bone `i`'s branch. An animation layer drives the bones from a branch's root
    up to its end.

    Returns:
        The next bone whose parent is not after `i`'s own (or the count of bones).
    """
    parent = bones[i]["parent"] if i else -1
    for j in range(i + 1, len(bones)):
        if bones[j]["parent"] <= parent:
            return j
    return len(bones)


# -- the game's list of clips --------------------------------------------------------------------


def clip_loops(clip_list: bytes) -> dict[str, tuple[int, int]]:
    """The first and last frame of each clip's loop for the leaders, from the markers in the game's list."""
    text = clip_list.decode("utf-8", "replace")
    loops: dict[str, tuple[int, int]] = {}
    for tag, clip, body in re.findall(r'\{\s*"([^"]+)"\s*"([^"]+)"\s*\d+\s*(.*?)\n\t\}', text, re.DOTALL):
        marks = [(float(a), int(b)) for a, b in re.findall(r'^\s*([0-9.]+) (\d+) "', body, re.MULTILINE)]
        start = [frame for frame, flag in marks if flag == LOOP_START]
        end = [frame for frame, flag in marks if flag == LOOP_END]
        if start and end and (tag.startswith("player_") or clip not in loops):
            loops[clip] = (int(start[0]), int(end[0]))
    return loops


def read_entries(clip_list: bytes) -> list[tuple[str, str, list[tuple[float, int, str]]]]:
    """
    The game's list's skeletal entries, in order: a tag's place in it is the number the game's code
    plays it by.

    Returns:
        [(tag, the clip it names, [(frame, flag, name)] of its markers)].

    Raises:
        BuildError: If the list is not laid out as expected.
    """
    head, _, rest = clip_list.partition(b"\r\n{\r\n")
    body, closed, _ = rest.partition(b"\r\n}\r\n")
    count = int(head.split(b"\r\n")[1].split()[0])
    tokens = []
    for line in body.split(b"\r\n"):
        line = line.strip()
        if line and not line.startswith(b"#"):
            tokens += re.findall(rb'"[^"]*"|\{|\}|[^\s{}"]+', line)
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] != b"{":
            raise BuildError(f"{CLIP_LIST}: expected an entry, found {tokens[i]!r}")
        tag, clip, n = tokens[i + 1].strip(b'"').decode(), tokens[i + 2].strip(b'"').decode(), int(tokens[i + 3])
        i += 4
        events = []
        for _ in range(n):
            events.append((float(tokens[i]), int(tokens[i + 1]), tokens[i + 2].strip(b'"').decode()))
            i += 3
            if tokens[i] == b"{":  # what an effect marker carries: skipped
                depth = 1
                while depth:
                    i += 1
                    depth += {b"{": 1, b"}": -1}.get(tokens[i], 0)
                i += 1
        if tokens[i] != b"}":
            raise BuildError(f"{CLIP_LIST}: entry {tag} does not end after its {n} markers")
        i += 1
        out.append((tag, clip, events))
    if not closed or len(out) != count:
        raise BuildError(f"{CLIP_LIST} counts {count} skeletal entries, and {len(out)} were read")
    return out


def tag_loop(clip_list: bytes, tag: str) -> tuple[str, int, int]:
    """
    The clip an entry of the game's list names, and the frames its loop runs between.

    Returns:
        The clip's name, and the first and last frames of its loop.

    Raises:
        BuildError: If there is no such entry, or it does not loop between two whole frames.
    """
    for name, clip, events in read_entries(clip_list):
        if name == tag:
            first = [frame for frame, flag, _ in events if flag == LOOP_START]
            last = [frame for frame, flag, _ in events if flag == LOOP_END]
            if (
                len(first) != 1
                or len(last) != 1
                or first[0] != int(first[0])
                or last[0] != int(last[0])
                or last[0] <= first[0]
            ):
                raise BuildError(f"the entry {tag} of {CLIP_LIST} does not loop between two whole frames")
            return clip, int(first[0]), int(last[0])
    raise BuildError(f"{CLIP_LIST} has no entry {tag}")


def define_stands(clip_list: bytes, stands: dict[str, tuple[Stand, tuple[int, int], int]]) -> bytes:
    """
    The game's list with each stand's entry naming his clip, and looping over all of it.

    The entry keeps its place, so the number the game plays it by is what it was, and nothing else
    of the list changes.

    Returns:
        The new list.

    Raises:
        BuildError: If an entry is not laid out as expected.
    """
    for name, (stand, loop, count) in stands.items():
        if loop[0] != 0:
            raise BuildError(f"the entry {stand.tag} does not loop from its first frame, which is not handled")
        old = f'\t\t"{stand.tag}" \r\n\t\t"{stand.like}" \r\n'.encode()
        if clip_list.count(old) != 1:
            raise BuildError(f"{CLIP_LIST} does not have the entry {stand.tag} naming {stand.like} once")
        head, _, rest = clip_list.partition(old)
        body, closed, tail = rest.partition(b"\r\n\t}")
        end = f'{float(loop[1]):.8f} {LOOP_END} "" '.encode()
        if not closed or body.count(end) != 1:
            raise BuildError(f"the entry {stand.tag} of {CLIP_LIST} is not laid out as expected")
        new = f'\t\t"{stand.tag}" \r\n\t\t"{name}" \r\n'.encode()
        clip_list = head + new + body.replace(end, f'{float(count):.8f} {LOOP_END} "" '.encode()) + closed + tail
    return clip_list


def move_marks(move: Move) -> list[tuple[float, int]]:
    """
    The markers a move's entry of the game's list has: one on the frame it lands, one on the frame
    the code acts on, and, for a move that goes round, the two the game's own looping entries have.

    Returns:
        (frame, flag) of each, in the order of their frames.
    """
    marks = [(move.hit, ACTION), (move.more, MORE)]
    if move.loop:
        marks += [(move.loop[0], LOOP_START), (move.loop[1], LOOP_END)]
    return sorted((float(frame), flag) for frame, flag in marks if frame is not None)


def define_moves(clip_list: bytes, moves: dict[str, Move]) -> tuple[bytes, int]:
    """
    The game's list with an entry for each move on the end of its skeletal section.

    A clip is played by its place in this list. Entries are only ever added at the end, so every
    number the game's own code plays a clip by still means what it did. Each move's tag is its
    clip's name.

    Returns:
        The new list, and the number of the first move in it.

    Raises:
        BuildError: If the list is not laid out as expected.
    """
    head, opened, rest = clip_list.partition(b"\r\n{\r\n")
    body, closed, tail = rest.partition(b"\r\n}\r\n")
    lines = head.split(b"\r\n")
    counts = lines[1].split() if len(lines) == 3 else []
    if not opened or not closed or not counts or not counts[0].isdigit():
        raise BuildError(f"{CLIP_LIST} is not laid out as expected")
    first = int(counts[0])
    added = ""
    for name, move in moves.items():
        marks = move_marks(move)
        added += f'\r\n\t# \r\n\t{{\r\n\t\t"{name}" \r\n\t\t"{name}" \r\n\t\t{len(marks)} '
        for frame, flag in marks:
            added += f'\r\n\t\t{frame:.8f} {flag} "" '
        added += "\r\n\t}"
    counts[0] = str(first + len(moves)).encode()
    lines[1] = b" ".join(counts) + b" "
    return b"\r\n".join(lines) + opened + body + added.encode() + closed + tail, first


# -- where his parts go, clip by clip --------------------------------------------------------------


def clip_scales(skel: Skeleton, clip: SkelAnim) -> dict[str, list[Vector]]:
    """{bone: its (x, y, z) size at each frame} for the leader's bones a clip stretches or squashes."""
    poses = [clip.pose(float(frame)) for frame in range(clip.frames + 1)]
    out = {}
    for b in skel.bones:
        sizes = [tuple(p.get(b["name"], {}).get("S", b["scale"])) for p in poses]
        if any(abs(c - 1.0) > 1e-4 for s in sizes for c in s):
            out[b["name"]] = sizes
    return out


def stock_world(skel: Skeleton, plan: Plan, pose: Pose, loose: tuple[str, ...] | list[str] = ()) -> list[Frame]:
    """
    The leader's own bones under a clip's pose, on the planned skeleton.

    A clip may stretch a bone named in `loose`: one that none of the leader's bones hangs under, and
    whose parts hang under a bone of his own that takes the stretch back out (`plan_own`). Its
    rotation is given here without the stretch, which is what those parts are then turned in.

    Returns:
        Where each bone sits, in the skeleton's order.

    Raises:
        BuildError: If the clip stretches any other bone.
    """
    out: list[Frame] = []
    for i in sorted(plan):
        b = skel.bones[i]
        p = pose.get(b["name"], {})
        scale = p.get("S", b["scale"])
        if any(abs(c - 1.0) > 1e-4 for c in scale) and b["name"] not in loose:
            raise BuildError(f"a clip scales bone {b['name']}; a part hung under it could not cancel that")
        rot = euler_matrix(*p.get("R", b["rot"])[:3])
        trans = p.get("T", plan[i][0])
        if b["parent"] >= 0:
            prot, porigin = out[b["parent"]]
            out.append((matmul(prot, rot), to_model(prot, porigin, trans)))
        else:
            out.append((rot, tuple(trans)))
    return out


def clip_targets(
    char: Character,
    clip: SkelAnim,
    loop: tuple[int, int],
    which: str,
    skel: Skeleton,
    plan: Plan,
    loose: list[str],
) -> tuple[int, list[dict[str, Target]]]:
    """
    For each frame of a leaders' clip, where his animation wants every part.

    His cycle is stretched over the clip's loop and repeated outside it, and slid in time so that
    his legs swing forward when the leaders' do: the footstep sounds and dust are timed to theirs.

    Returns:
        The frame of his cycle that the loop starts on, and the targets frame by frame.
    """
    link = char.link
    anim = link.animation(which)
    names = [p.name for p in link.parts]
    first, last = loop
    span = float(last - first)
    stock = [stock_world(skel, plan, clip.pose(float(frame)), loose) for frame in range(clip.frames + 1)]
    bone = {b["name"]: i for i, b in enumerate(skel.bones)}

    def tau(frame: int, phase: int) -> float:
        return phase + (frame - first) * anim.frames / span

    def poses(t: float) -> list[Frame]:
        return link.pose(anim.rotations(t), anim.root(t))

    def centred(row: list[float]) -> list[float]:
        return [v - sum(row) / len(row) for v in row]

    # how far forward each leg points, theirs per clip frame and his per frame of his own cycle
    theirs = [[stock[frame][bone[b]][0][2][0] for frame in range(first, last)] for b in ("llegjnt", "rlegjnt")]
    his = []
    for side in ("left", "right"):
        row = []
        for t in range(anim.frames):
            p = poses(float(t))
            row.append(p[names.index(side + "_foot")][1][2] - p[names.index(side + "_thigh")][1][2])
        his.append(row)
    theirs, his = [centred(r) for r in theirs], [centred(r) for r in his]

    def score(phase: int) -> float:
        return sum(
            theirs[s][frame - first] * his[s][round(tau(frame, phase)) % anim.frames]
            for s in range(2)
            for frame in range(first, last)
        )

    phase = max(range(anim.frames), key=score)
    frames = []
    for frame in range(clip.frames + 1):
        p = poses(tau(frame, phase))
        row = {}
        for part in char.frames:
            rot, pos = p[names.index(part)]
            prot, porigin = stock[frame][bone[char.stock[part]]]
            row[part] = (prot, porigin, rot, char.place(pos))
        frames.append(row)
    return phase, frames


def stand_targets(
    char: Character,
    stand: Stand,
    clip: SkelAnim,
    loop: tuple[int, int],
    skel: Skeleton,
    plan: Plan,
) -> tuple[int, list[Pose], list[dict[str, Target]]]:
    """
    For each frame of a stand, the stock clip's pose for the leader's bones, and where his own
    animations want every part.

    The clip is a whole number of the stock loop's rounds long, the number that brings his own
    sequence nearest the speed `stand.rate` asks, so both come round together at its end.

    Returns:
        The clip's last frame, the stock clip's pose on each frame, and the targets frame by frame.
    """
    link = char.link
    first, last = loop
    period = last - first
    seq = link.sequence(stand.pieces)
    count = max(1, round(seq.frames * stand.rate / period)) * period
    names = [p.name for p in link.parts]
    bone = {b["name"]: i for i, b in enumerate(skel.bones)}
    seen, poses, frames = {}, [], []
    for frame in range(count + 1):
        k = first + frame % period
        if k not in seen:
            pose = clip.pose(float(k))
            seen[k] = (pose, stock_world(skel, plan, pose))
        pose, stock = seen[k]
        t = frame * seq.frames / float(count)
        p = link.pose(seq.turns(t), seq.root(t))
        row = {}
        for part in char.frames:
            rot, pos = p[names.index(part)]
            prot, porigin = stock[bone[char.stock[part]]]
            row[part] = (prot, porigin, rot, char.place(pos))
        poses.append(pose)
        frames.append(row)
    return count, poses, frames


def stand_tracks(clip: SkelAnim, poses: list[Pose]) -> list[Track]:
    """
    A stand's tracks for the leader's own bones: the stock clip's, laid out over the stand's frames.

    Returns:
        One track for each bone the stock clip moves, in its order.

    Raises:
        BuildError: If the stock clip leaves a bone's size or rotation to its model.
    """
    tracks = []
    for name, (base, _) in clip.bones().items():
        if "S" not in base or "R" not in base:
            raise BuildError(f"clip {clip.name} gives bone {name} no size or no rotation, which is not handled")
        curves = {}
        for key, targets in (("S", SCALE_CURVES), ("R", ROTATE_CURVES), ("T", TRANSLATE_CURVES)):
            if key in base:
                for k, target in enumerate(targets):
                    values = [pose[name][key][k] for pose in poses]
                    if max(values) - min(values) > 1e-9:
                        curves[target] = values
        at = poses[0][name]
        tracks.append((name, tuple(at["S"]), tuple(at["R"]), tuple(at["T"]) if "T" in base else None, curves))
    return tracks


def move_targets(char: Character, move: Move, skel: Skeleton, plan: Plan) -> tuple[Pose, list[dict[str, Target]]]:
    """
    For each frame of a move, where his animation wants every part, and where the leader's bones are.

    The leader's own bones rest, but for the bone the punch draws its spark on, which is moved so
    that the spark's point is on the blade: on the frame a swing lands, or on every frame of the
    spin on which the blade can hit (the bone then follows the blade over those frames, and holds
    still before and after). A move with no spark leaves that bone at rest too.

    Returns:
        What the clip tells the leader's own bones, and the targets frame by frame.
    """
    names = [p.name for p in char.link.parts]
    bone = {b["name"]: i for i, b in enumerate(skel.bones)}
    poses = char.move_poses(move)

    def placed(at: int) -> Pose:
        """The spark's bone, put so that its point is where the spark belongs on frame `at`."""
        point = char.move_point(poses[at])
        origin = to_model(rest[i][0], point, (-HIT_REACH, 0.0, 0.0))
        return {HIT_BONE: {"T": list(to_local(rest[parent][0], rest[parent][1], origin))}}

    each = None
    if move.spark is None:
        pose = {}
    else:
        rest = stock_world(skel, plan, {})
        i = bone[HIT_BONE]
        parent = skel.bones[i]["parent"]
        at = move.spark[1]
        if isinstance(at, tuple):
            each = [placed(min(max(k, at[0]), at[1])) for k in range(len(poses))]
            pose = {HIT_BONE: {"T": [p[HIT_BONE]["T"] for p in each]}}
        else:
            pose = placed(at)
    stock = stock_world(skel, plan, pose) if each is None else None
    frames = []
    for k, p in enumerate(poses):
        row = {}
        under = stock if each is None else stock_world(skel, plan, each[k])
        for name in char.frames:
            rot, pos = p[names.index(name)]
            prot, porigin = under[bone[char.stock[name]]]
            row[name] = (prot, porigin, rot, char.place(pos))
        frames.append(row)
    return pose, frames


def move_rest(skel: Skeleton, plan: Plan, pose: Pose) -> list[Track]:
    """
    A move's tracks for the leader's own bones: every one at rest, as the model has it, but for what
    `pose` says. As in the stock clips, only the root (and here the spark's bone) is told where to
    be. A place given for every frame (a spark's bone that follows the blade) is a curve of its three
    numbers, the first frame's the constant.

    Returns:
        One track for each bone, in the skeleton's order.
    """
    tracks = []
    for i, b in enumerate(skel.bones):
        trans = pose.get(b["name"], {}).get("T", plan[i][0] if b["parent"] < 0 else None)
        curves = {}
        if trans is not None and isinstance(trans[0], (list, tuple)):
            curves = {target: [float(t[k]) for t in trans] for k, target in enumerate(TRANSLATE_CURVES)}
            trans = trans[0]
        tracks.append(
            (b["name"], (1.0, 1.0, 1.0), tuple(b["rot"][:3]), None if trans is None else tuple(trans), curves),
        )
    return tracks


# -- his bones -------------------------------------------------------------------------------------


def _quarter_turns() -> list[Matrix]:
    """The 24 ways to turn a cube onto itself."""
    seen, out = set(), []
    for a in range(4):
        for b in range(4):
            for c in range(4):
                m = euler_matrix(a * math.pi / 2, b * math.pi / 2, c * math.pi / 2)
                m = [[float(round(v)) for v in row] for row in m]
                key = tuple(tuple(row) for row in m)
                if key not in seen:
                    seen.add(key)
                    out.append(m)
    return out


def choose_turn(samples: list[Matrix]) -> Matrix:
    """
    A fixed turn for a part's bone that keeps its angles away from the pole.

    Three angles cannot describe a rotation smoothly when the middle one nears 90 degrees: the other
    two swing wildly from frame to frame, and the game, which blends between frames, would show it.
    Which rotations land there depends on how the bone's own axes are set, and that is free: the
    mesh is bound to whatever they are. `samples` are the rotations the bone will be asked for.

    Returns:
        The quarter turn that keeps the samples furthest from the pole.
    """
    best = None
    for turn in _quarter_turns():
        worst = max(abs(matmul(rot, turn)[2][0]) for rot in samples)
        if best is None or worst < best[0] - 1e-9:
            best = (worst, turn)
    return best[1]


def along_bone(turn: Matrix, size: Vector) -> Vector:
    """
    A size along the three axes of a part, along its bone's: the bone's axes are the part's turned by
    `turn`, a whole number of quarter turns, so each of its axes lies along one of the part's.

    Returns:
        The size along the bone's three axes.
    """
    return tuple(size[[abs(turn[r][j]) > 0.5 for r in range(3)].index(True)] for j in range(3))


def plan_own(
    char: Character,
    skel: Skeleton,
    world: list[Frame],
    plan: Plan,
    targets: dict[str, list[dict[str, Target]]],
    scales: dict[str, dict[str, list[Vector]]],
    shows: dict[str, list[tuple[str, ...]]],
    sized: dict[str, dict[str, list[Vector]]],
) -> tuple[list[tuple], dict[str, list[Track]], list[Track], dict[int, str]]:
    """
    His own bones, and their tracks for every clip in `targets`.

    `scales` gives, for a clip, the size it gives a leader's bone at each frame. Such a bone is given
    a bone of his own between it and his parts (`m_` and its name), which the same clip gives the
    opposite size, so that the parts turn in a space that is not stretched. In every other clip it
    rests and changes nothing.

    A look (`char.looks`: {name: the part it is on}) is one of several things a part can be drawn
    as. Each has a bone of its own under its part's, at rest exactly on it, and a clip gives it its
    size and nothing else: one or `HIDDEN`, by what `shows` says shows on each frame (the sword and
    shield out where it says nothing), or, from `sized`, a size along each of its part's axes (the
    ring's, which grows and thins).

    The points of his sword's trail (`char.trail["slots"]`) have bones under the bone of the look
    that shows the trail, at rest where the parts rest, and tracks in the clips that show it
    (`char.trail["clips"]`) and in no other: a track costs memory in every Pikmin's and leader's
    animations for every clip it is in. In every other clip nothing moves them, and the look keeps
    them small.

    The parts that follow the leader's legs go in the run of bones the game's legs layer drives: the
    bone named `legcentre` and the rest of its branch. A bone can only be added at the end of the
    list, so the leader's own three are renamed, which leaves them out of every clip, and copies
    under their names are put after the upper body's new bones, with the legs' parts after those.
    The copy of `legcentre` hangs under the root, which is what ends the upper body's run there.

    Returns:
        The bones as `Grower.add_bones` takes them, each clip's tracks for them, the tracks that hold
        them at rest in every other clip, and the leader's bones to rename.
    """
    names = [b["name"] for b in skel.bones]
    bones, still, turns = [], [], {}
    tracks: dict[str, list[Track]] = {clip: [] for clip in targets}
    at = {name: i for i, name in enumerate(names)}  # where each of the leader's bones is now
    renames, loose, under_index = {}, {}, {}
    looks, where = char.looks, {}  # (`where`: the index of each part's bone, and of each look's)
    legs = range(names.index(LEGS_ROOT), branch_end(skel.bones, names.index(LEGS_ROOT)))
    stretched = sorted({bone for sizes in scales.values() for bone in sizes}, key=names.index)

    def under(leader: str) -> int:
        """The index his parts hang under for this leader bone, adding the bone that takes its stretch out."""
        if leader not in stretched:
            return at[leader]
        if leader not in loose:
            name = loose[leader] = OWN + leader
            under_index[leader] = len(names) + len(bones)
            nothing = (name, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), {})
            bones.append((name, at[leader], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False))
            still.append(nothing)
            for clip in targets:
                sizes = scales.get(clip, {}).get(leader)
                if sizes is None:
                    tracks[clip].append(nothing)
                    continue
                back = [tuple(1.0 / c for c in size) for size in sizes]
                curves = {target: [size[k] for size in back] for k, target in enumerate(SCALE_CURVES)}
                curves = {t: v for t, v in curves.items() if max(v) - min(v) > 1e-6}
                tracks[clip].append((name, back[0], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), curves))
        return under_index[leader]

    def own(part: str) -> None:
        stock = names.index(char.stock[part])  # the leader's bone as planned, for where things are
        parent = under(char.stock[part])
        prot, porigin = world[stock][0], plan[stock][1]
        rot0, origin = char.frames[part]
        samples = [matmul(transpose(prot), rot0)]
        for frames in targets.values():
            samples += [matmul(transpose(row[part][0]), row[part][2]) for row in frames]
        turn = choose_turn(samples)
        angles = matrix_euler(matmul(samples[0], turn))
        trans = to_local(prot, porigin, origin)
        where[part] = len(names) + len(bones)
        bones.append((OWN + part, parent, angles, trans, True))
        still.append((OWN + part, (1.0, 1.0, 1.0), angles, trans, {}))
        turns[part] = turn
        for clip, frames in targets.items():
            rots, places, near = [], [], angles
            for p_rot, p_origin, want_rot, want_pos in (row[part] for row in frames):
                near = matrix_euler(matmul(matmul(transpose(p_rot), want_rot), turn), near)
                rots.append(near)
                places.append(to_local(p_rot, p_origin, want_pos))
            sizes = [1.0] * len(frames)
            curves = {target: [r[k] for r in rots] for k, target in enumerate(ROTATE_CURVES)}
            curves.update({target: [t[k] for t in places] for k, target in enumerate(TRANSLATE_CURVES)})
            curves.update(dict.fromkeys(SCALE_CURVES, sizes))
            curves = {t: v for t, v in curves.items() if max(v) - min(v) > 1e-6}
            tracks[clip].append((OWN + part, (sizes[0],) * 3, rots[0], places[0], curves))

    def look(name: str) -> None:
        """A look's bone: under its part's, on it, and in every clip its size and nothing else."""
        usual = 1.0 if name in char.shown else HIDDEN
        nothing = (0.0, 0.0, 0.0)
        where[name] = len(names) + len(bones)
        bones.append((OWN + name, where[looks[name]], nothing, nothing, True))
        still.append((OWN + name, (usual,) * 3, nothing, nothing, {}))
        for clip, frames in targets.items():
            given = sized.get(clip, {}).get(name)
            if given is not None:
                given = [along_bone(turns[looks[name]], size) for size in given]
                curves = {target: [size[k] for size in given] for k, target in enumerate(SCALE_CURVES)}
                curves = {t: v for t, v in curves.items() if max(v) - min(v) > 1e-6}
                tracks[clip].append((OWN + name, tuple(given[0]), nothing, nothing, curves))
                continue
            seen = shows.get(clip) or [char.shown] * len(frames)
            sizes = [1.0 if name in shown else HIDDEN for shown in seen]
            curves = dict.fromkeys(SCALE_CURVES, sizes) if max(sizes) - min(sizes) > 1e-6 else {}
            tracks[clip].append((OWN + name, (sizes[0],) * 3, nothing, nothing, curves))

    def slot(name: str) -> None:
        """
        A point of the trail's bone: under the bone of the look that shows the trail, which rests on its
        part's, at rest where the point's part rests, and, in the clips that show the trail, where the
        part is on every frame, against where the look's part is then.
        """
        part = looks[trail["look"]]
        p_rot0, p_origin0 = matmul(char.frames[part][0], turns[part]), char.frames[part][1]
        rot0, origin = char.frames[name]
        samples = [matmul(transpose(p_rot0), rot0)]
        for clip in shown_in:
            samples += [matmul(transpose(matmul(row[part][2], turns[part])), row[name][2]) for row in targets[clip]]
        turn = choose_turn(samples)
        angles = matrix_euler(matmul(samples[0], turn))
        bones.append((OWN + name, where[trail["look"]], angles, to_local(p_rot0, p_origin0, origin), True))
        for clip in shown_in:
            rots, places, near = [], [], angles
            for row in targets[clip]:
                p_rot = matmul(row[part][2], turns[part])
                near = matrix_euler(matmul(matmul(transpose(p_rot), row[name][2]), turn), near)
                rots.append(near)
                places.append(to_local(p_rot, row[part][3], row[name][3]))
            curves = {target: [r[k] for r in rots] for k, target in enumerate(ROTATE_CURVES)}
            curves.update({target: [t[k] for t in places] for k, target in enumerate(TRANSLATE_CURVES)})
            curves = {t: v for t, v in curves.items() if max(v) - min(v) > 1e-6}
            tracks[clip].append((OWN + name, (1.0, 1.0, 1.0), rots[0], places[0], curves))

    trail = char.trail
    shown_in = [clip for clip in targets if clip in trail["clips"]]
    parts = sorted((p for p in char.frames if p not in trail["slots"]), key=lambda p: (names.index(char.stock[p]), p))
    for part in parts:
        if names.index(char.stock[part]) not in legs:
            own(part)
    for name in sorted(looks):  # (in the run of bones their parts are in)
        if names.index(char.stock[looks[name]]) not in legs:
            look(name)
    for name in trail["slots"]:  # (after the look they hang under, before the legs' layer)
        slot(name)
    for i in legs:
        b = skel.bones[i]
        renames[i] = OLD + b["name"]
        bones.append((b["name"], at[names[b["parent"]]], tuple(b["rot"][:3]), tuple(plan[i][0]), False))
        at[b["name"]] = len(names) + len(bones) - 1
    for part in parts:
        if names.index(char.stock[part]) in legs:
            own(part)
    for name in sorted(looks):
        if names.index(char.stock[looks[name]]) in legs:
            look(name)
    return bones, tracks, still, renames


# -- textures --------------------------------------------------------------------------------------


def _sample(image: Image, u: float, v: float, *, alpha: bool = False) -> tuple[int, int, int, int]:
    """
    A bilinear sample of an image at a texture coordinate (v up), clamped at the edges.

    Returns:
        The pixel, opaque unless `alpha` keeps the picture's own alpha.
    """
    w, h, rows = image
    x = min(max(u * w - 0.5, 0.0), w - 1.0)
    y = min(max((1.0 - v) * h - 0.5, 0.0), h - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = x - x0, y - y0
    a, b, c, e = rows[y0][x0], rows[y0][x1], rows[y1][x0], rows[y1][x1]

    def mix(i: int) -> int:
        return round((a[i] * (1 - fx) + b[i] * fx) * (1 - fy) + (c[i] * (1 - fx) + e[i] * fx) * fy)

    return (*(mix(i) for i in range(3)), mix(3) if alpha else 255)


class Atlas:
    """One image holding every material a shape uses, and the mapping from the old coordinates."""

    def __init__(self, char: Character, tris: list[list[Corner]], width: int, height: int) -> None:
        """
        Pack and draw the materials of `tris` into a `width` by `height` image.

        A material that is a cut-out keeps its picture's alpha, for a shape whose material tests
        alpha: his collar and earrings.

        Raises:
            BuildError: If they do not fit even at their own size.
        """
        self.width, self.height = width, height
        images, used = {}, {}
        for tri in tris:
            for _, _, uv, material, _ in tri:
                if material not in images:
                    images[material] = char.image(material)
                r = used.setdefault(material, [uv[0], uv[0], uv[1], uv[1]])
                r[0], r[1], r[2], r[3] = min(r[0], uv[0]), max(r[1], uv[0]), min(r[2], uv[1]), max(r[3], uv[1])
        self.cut = {material for material in images if material.cutout}
        self.solid = {}
        for material, (_, _, rows) in images.items():
            first = tuple(rows[0][0][:3])
            if material not in self.cut and all(tuple(p[:3]) == first for row in rows for p in row):
                self.solid[material] = first
        self.tiles = {}
        for zoom in char.zooms:
            if self._pack(images, used, zoom):
                break
        else:
            raise BuildError(f"the materials do not fit a {width}x{height} texture")
        self.zoom = zoom
        background = (*next(iter(self.solid.values()), (128, 128, 128)), 255)
        self.rows = [[background] * width for _ in range(height)]
        for material, (x, y, w, h, u0, u1, v0, v1) in self.tiles.items():
            for py in range(h):
                v = v1 - (py + 0.5) / h * (v1 - v0)
                row = self.rows[y + py]
                for px in range(w):
                    if material in self.solid:
                        row[x + px] = (*self.solid[material], 255)
                    else:
                        row[x + px] = _sample(
                            images[material],
                            u0 + (px + 0.5) / w * (u1 - u0),
                            v,
                            alpha=material in self.cut,
                        )

    def _pack(self, images: dict[Material, Image], used: dict[Material, list[float]], zoom: float) -> bool:
        """Shelf-pack the tiles, tallest first; False if they overflow at this zoom."""
        want = []
        for material, (w, h, _) in images.items():
            if material in self.solid:
                want.append((SOLID_TILE, SOLID_TILE, material.name, material, 0.0, 1.0, 0.0, 1.0))
                continue
            u0, u1, v0, v1 = used[material]
            u0, u1, v0, v1 = u0 - MARGIN / w, u1 + MARGIN / w, v0 - MARGIN / h, v1 + MARGIN / h
            want.append(
                (
                    math.ceil((u1 - u0) * w * zoom),
                    math.ceil((v1 - v0) * h * zoom),
                    material.name,
                    material,
                    u0,
                    u1,
                    v0,
                    v1,
                ),
            )
        self.tiles = {}
        x = y = GUTTER
        shelf = 0
        for tw, th, _, material, u0, u1, v0, v1 in sorted(want, key=lambda t: (-t[1], t[2])):
            if x + tw + GUTTER > self.width:
                x, y, shelf = GUTTER, y + shelf + GUTTER, 0
            if x + tw + GUTTER > self.width or y + th + GUTTER > self.height:
                return False
            self.tiles[material] = (x, y, tw, th, u0, u1, v0, v1)
            x += tw + GUTTER
            shelf = max(shelf, th)
        return True

    def map(self, material: Material, uv: tuple[float, float]) -> tuple[float, float]:
        """An old coordinate (v up) as the atlas's (v down, as the game's textures are addressed)."""
        x, y, w, h, u0, u1, v0, v1 = self.tiles[material]
        if material in self.solid:
            return ((x + w / 2) / self.width, (y + h / 2) / self.height)  # one point: never blurs or bleeds
        return ((x + (uv[0] - u0) / (u1 - u0) * w) / self.width, (y + (v1 - uv[1]) / (v1 - v0) * h) / self.height)


class Given:
    """What the ring's corners read of its texture: the coordinates it gives, as they are (v down)."""

    @staticmethod
    def map(_material: Material, uv: tuple[float, float]) -> tuple[float, float]:
        """The coordinate as given."""
        return (uv[0], uv[1])


# -- one shape -------------------------------------------------------------------------------------


def weld(tris: list[list[Corner]], atlas: Atlas | Given, palette: dict[str, int]) -> tuple[list[tuple], list[int]]:
    """
    Turn triangles of corners into vertices and indices.

    Returns:
        The vertices, each (position, normal, uv, matrix number, part), and three indices a triangle.
    """
    verts, index, indices = [], {}, []
    for tri in tris:
        for pos, nrm, uv, material, part in tri:
            vert = (pos, nrm, atlas.map(material, uv), palette[part], part)
            key = (
                tuple(round(c, 4) for c in pos),
                tuple(round(c, 3) for c in nrm),
                tuple(round(c, 5) for c in vert[2]),
                part,
            )
            if key not in index:
                index[key] = len(verts)
                verts.append(vert)
            indices.append(index[key])
    return verts, indices


def write_shape(shape: Shape, verts: list[tuple], indices: list[int], frames: dict[str, Frame]) -> None:
    """
    Overwrite a shape's vertices, triangles and bounds. `frames` is where each part's bone rests.

    Raises:
        BuildError: If the new mesh does not fit the shape's buffers.
    """
    if len(verts) > shape.nvert:
        raise BuildError(f"{shape.name}: {len(verts)} vertices do not fit in {shape.nvert}")
    tangents, bitangents = [], []
    for _, nrm, _, _, _ in verts:
        # The normal map is flat, so any frame square to the normal lights the same.
        tangent = unit(cross((0.0, 1.0, 0.0), nrm)) if abs(nrm[1]) < 0.99 else (1.0, 0.0, 0.0)
        tangents.append(tangent)
        bitangents.append(unit(cross(nrm, tangent)))
    values = {
        "_p0": [v[0] for v in verts],
        "_n0": [v[1] for v in verts],
        "_t0": tangents,
        "_b0": bitangents,
        "_c0": [(1.0, 1.0, 1.0, 1.0)] * len(verts),
        "_u0": [v[2] for v in verts],
        "_i0": [(v[3], v[3]) for v in verts],
        "_w0": [(1.0, 0.0)] * len(verts),
    }
    for name in shape.attrs:
        if name not in values:
            raise BuildError(f"{shape.name}: no rule for vertex attribute {name}")
        shape.write(name, values[name])
    shape.set_indices(indices)
    # Bounds, measured the way the stock ones are: each vertex in the space of the bone it follows.
    local = [to_local(frames[v[4]][0], frames[v[4]][1], v[0]) for v in verts]
    lo = [min(p[i] for p in local) for i in range(3)]
    hi = [max(p[i] for p in local) for i in range(3)]
    centre = [(lo[i] + hi[i]) / 2 for i in range(3)]
    radius = max(math.sqrt(dot(sub(p, centre), sub(p, centre))) for p in local)
    shape.set_bounds(centre, [(hi[i] - lo[i]) / 2 for i in range(3)], radius)


def shape_of(model: Model, materials: tuple[str, ...], stock: Model | None = None) -> Shape:
    """
    The first shape the stock model (`stock`, or this one) draws with one of `materials`.

    Returns:
        That shape of this model.

    Raises:
        BuildError: If the model has no such shape.
    """
    ref = stock or model
    for i, shape in enumerate(ref.shapes):
        if ref.mats[shape.mat].name in materials:
            return model.shapes[i]
    raise BuildError(f"{model.name} has no shape with material {' or '.join(materials)}")


def split(char: Character) -> tuple[list[list[Corner]], list[list[Corner]]]:
    """
    His triangles for the body shape and for the head shape.

    The head shape follows the head's bone alone in the stock model, so it takes the triangles that
    do, all three corners; and it is the body's material that tests alpha, so a cut-out goes to the
    body wherever it is. The looks his model names for the head's shape go there too, though they
    follow other bones: the body's shape has only so many vertices, and the head's has room.

    Returns:
        The body's triangles, and the head's.
    """
    body, head = [], []
    for tri in char.tris:
        material = tri[0][3]
        # (a look named for the head's shape may have a material that only says it is a cut-out: what counts there is
        # whether its picture has anything to cut)
        whole = (all(char.stock[c[4]] == "headjnt" for c in tri) and not material.cutout) or (
            all(c[4] in char.in_head for c in tri) and not see_through(char, material)
        )
        (head if whole else body).append(tri)
    return body, head


def see_through(char: Character, material: Material) -> bool:
    """Whether an alpha test drops anything of a material: it is a cut-out, and some pixel is mostly see-through."""
    return material.cutout and any(px[3] < 128 for row in char.image(material)[2] for px in row)


# -- the build -------------------------------------------------------------------------------------


def remodel(
    stock: bytes,
    char: Character,
    clip_list: bytes,
) -> tuple[bytes, dict[str, tuple[Stand, tuple[int, int], int]]]:
    """
    The leaders' model file with Alph rebuilt as Link.

    `clip_list` is the game's own list of clips, for the loops the leaders play them in.

    Returns:
        The new model file, and his stands: {clip: (stand, the stock entry's loop, its last frame)}.

    Raises:
        BuildError: If the stock model is not put together the way this expects.
    """
    f0 = Bfres(stock)
    model0 = f0.model("player" + LEADER)
    skel0 = model0.skel
    names0 = [b["name"] for b in skel0.bones]
    world0, plan = plan_skeleton(skel0, char)
    body_tris, head_tris = split(char)
    stock_clips = {clip.name: clip for clip in f0.skel_anims()}

    # Where his animations put each part, in every clip that will carry curves for his bones.
    targets, scales = {}, {}
    loops = clip_loops(clip_list)
    for name in char.clips:
        found = clip_scales(skel0, stock_clips[name])
        if found:
            scales[name] = found
    # A bone a clip stretches is no trouble when nothing of the leader's hangs under it: his parts
    # under it are given a bone that takes the stretch out.
    loose = sorted({bone for found in scales.values() for bone in found}, key=names0.index)
    for bone in loose:
        if any(b["parent"] == names0.index(bone) for b in skel0.bones):
            raise BuildError(f"a clip scales bone {bone}, which other bones hang under; a part could not cancel that")
    for name, which in char.clips.items():
        clip = stock_clips[name]
        loop = loops.get(name, (0, clip.frames))
        phase, targets[name] = clip_targets(char, clip, loop, which, skel0, plan, loose)
        logger.info("  clip %-8s his %s, from frame %d of it", name, which, phase)
    stands, standing = {}, {}
    for name, stand in char.stands.items():
        clip, first, last = tag_loop(clip_list, stand.tag)
        if clip != stand.like:
            raise BuildError(f"the entry {stand.tag} of {CLIP_LIST} names {clip}, not {stand.like}")
        count, standing[name], targets[name] = stand_targets(char, stand, stock_clips[clip], (first, last), skel0, plan)
        stands[name] = (stand, (first, last), count)
        logger.info("  stand %-7s %3d frames: %s", name, count + 1, " + ".join(stand.pieces))
    poses, shows, sized = {}, {}, {}
    for name, move in char.moves.items():
        poses[name], targets[name] = move_targets(char, move, skel0, plan)
        if move.shows:
            shows[name] = move.shows
        if move.sizes:
            sized[name] = move.sizes
        logger.info("  move %-12s %3d frames: %s %s", name, move.count + 1, move.family, move.kind)
    bones, tracks, still, renames = plan_own(char, skel0, world0, plan, targets, scales, shows, sized)

    # Grow the file: his bones, a track for each in every clip, then the new clips.
    grower = bfres_edit.Grower(stock)
    first = grower.add_bones(skel0.off, bones, renames)[0]
    index = {name: first + k for k, (name, *_) in enumerate(bones)}
    body0, head0 = shape_of(model0, BODY_MATERIALS), shape_of(model0, HEAD_MATERIALS)
    glows = (char.glow, char.streak)  # the ring of his spin, and the trail of his sword
    drawn = [(body0, body_tris), (head0, head_tris)]
    drawn += [(shape_of(model0, (glow["material"],)), glow["tris"]) for glow in glows]
    for shape, tris in drawn:
        grower.set_skin_bones(shape.off, sorted({index[OWN + c[4]] for tri in tris for c in tri}))
    for clip in f0.skel_anims():
        grower.add_tracks(clip.o, tracks.get(clip.name, still))
    # Every change to a stock clip is made; from here the clip list is a copy with his clips on its end.
    # Each is modelled on a stock clip (a move plays once or goes round, a stand round and round) and
    # holds, for the leader's own bones, what that kind of clip must: rest for a move, the stock clip for a stand.
    offset = {clip.name: clip.o for clip in f0.skel_anims()}
    added = [(name, offset[move.like], move_rest(skel0, plan, poses[name])) for name, move in char.moves.items()]
    added += [
        (name, offset[stand.like], stand_tracks(stock_clips[stand.like], standing[name]))
        for name, (stand, _, _) in stands.items()
    ]
    new_clips = grower.add_clips([(name, len(targets[name]) - 1, like) for name, like, _ in added])
    for clip_offset, (name, _, theirs) in zip(new_clips, added, strict=True):
        grower.add_tracks(clip_offset, theirs + tracks[name])
    data = bytearray(grower.finish())
    logger.info(
        "  %d bones and %d clips added; the model file grows by %d bytes",
        len(bones),
        len(added),
        len(data) - len(stock),
    )

    # Then, in place: the leader's joints, the mesh and the textures.
    f = Bfres(data)
    model = f.model("player" + LEADER)
    skel, world = write_skeleton(f, model.skel, plan)
    names = [b["name"] for b in skel.bones]
    palette, frames = {}, {}
    for part in list(char.frames) + sorted(char.looks):
        bone = names.index(OWN + part)
        if bone not in skel.matrix_bones[: skel.nsmooth]:
            raise BuildError(f"bone {names[bone]} has no skinning matrix in {model.name}")
        palette[part] = skel.matrix_bones.index(bone)
        frames[part] = world[bone]
    body, head = shape_of(model, BODY_MATERIALS, model0), shape_of(model, HEAD_MATERIALS, model0)
    textures = bntx.Bntx(data)
    for shape, tris in ((body, body_tris), (head, head_tris)):
        material = model.mats[shape.mat]
        unskinned = {OWN + c[4] for tri in tris for c in tri} - {names[b] for b in shape.skinbones}
        if unskinned:
            raise BuildError(f"{shape.name} is not skinned to {', '.join(sorted(unskinned))}")
        colour = textures[material.role["_a0"]]
        atlas = Atlas(char, tris, colour.width, colour.height)
        verts, indices = weld(tris, atlas, palette)
        write_shape(shape, verts, indices, frames)
        bntx.replace(data, colour, atlas.rows)
        for role, value in (("_n0", FLAT_NORMAL), ("_s0", SPECULAR), ("_r0", ROUGHNESS)):
            if role in material.role:
                bntx.fill(data, textures[material.role[role]], value)
        logger.info(
            "  %-22s %d triangles, %d vertices, %d materials at x%g",
            shape.name,
            len(tris),
            len(verts),
            len(atlas.tiles),
            atlas.zoom,
        )
    # The ring of his spin and the trail of his sword: each in a shape of a material of the leaders' that draws a
    # picture, told to draw see-through, and given a picture of its own over a texture of the model's that nothing
    # drawn uses, which the material is pointed at by name. The trail's material is first made to draw as the ring's
    # does: given its shader options (which pick the shader archive's program that draws it) and its numbers, as the
    # stock model has them, its own over those; and the animations that would swap its pictures swap in its own alone.
    kept = [body, head]
    for glow, what in zip(glows, ("the ring of his spin", "the trail of his sword"), strict=True):
        shape = shape_of(model, (glow["material"],), model0)
        material = model.mats[shape.mat]
        verts, indices = weld(glow["tris"], Given(), palette)
        write_shape(shape, verts, indices, frames)
        if glow.get("like"):
            like = next(m for m in model0.mats if m.name == glow["like"])
            for name, (_, value) in sorted(like.options().items()):
                if material.options()[name][1] != value:
                    material.set_option(name, value)
            for name in like.params():
                if material.param(name) != like.param(name):
                    material.set_param(name, like.param(name))
        for name, values in sorted(glow["params"].items()):
            material.set_param(name, values)
        for name, value in sorted(glow["render"].items()):
            material.set_render(name, value)
        slot, name, rows = glow["texture"]
        material.set_texture(slot, name)
        bntx.replace(data, textures[name], rows)
        for anim in glow.get("anims", ()):
            for _, at in f.mat_anim_textures(anim):
                f.set_mat_anim_texture(at, name)
        logger.info("  %-22s %d triangles, %d vertices: %s", shape.name, len(glow["tris"]), len(verts), what)
        kept.append(shape)
    for other in model.shapes:
        if not any(other is shape for shape in kept):
            other.set_indices([])
    return bytes(data), stands


def hide_lights(ptcl: bytes) -> bytes:
    """
    The effect file with the lights of the leader's antenna made invisible.

    Returns:
        The new effect file, the same size as the old.

    Raises:
        BuildError: If the effects are not found, or do not look like what is expected.
    """
    data = bytearray(ptcl)
    wanted = [name.format(LEADER) for name in LIGHTS]
    for _, name, emitters in vfxb.emitter_blocks(data):
        if name not in wanted:
            continue
        wanted.remove(name)
        if not emitters:
            raise BuildError(f"effect {name} has no emitters")
        for emitter, block in emitters:
            colours = struct.unpack_from("<8f", data, block + vfxb.EMITTER_COLOR0)
            size = struct.unpack_from("<3f", data, block + vfxb.EMITTER_SCALE)
            # Refuse anything that does not hold colours and a size where a light's emitter does.
            if not (all(0.0 <= c <= 4.0 for c in colours) and all(0.5 <= x <= 500.0 for x in size)):
                raise BuildError(f"effect {name} / {emitter} does not hold colours and a size where expected")
            vfxb.hide_emitter(data, block)
    if wanted:
        raise BuildError(f"effects not found: {', '.join(wanted)}")
    return bytes(data)


class Built:
    """What the model step hands on to the steps after it: where his clips are in the game's list."""

    def __init__(self, entries: list[tuple[str, str, list]], first: int, moves: dict[str, Move]) -> None:
        self.first = first  # the number of his first clip in the game's list
        self.numbers = {name: first + k for k, name in enumerate(moves)}
        self.stand = next(k for k, (tag, _, _) in enumerate(entries) if tag == "player_wait")  # the leaders' stand


def build(char: Character, romfs: RomFs, mod: Mod) -> Built:
    """
    Write the model, the list of clips and the effects into the mod folder.

    Returns:
        Where his clips are in the game's list.
    """
    archive, settings, effects = romfs.read(ARCHIVE), romfs.read(SETTINGS), romfs.read(EFFECTS)
    clip_list = sarc.Sarc(yaz0.decompress(settings)).files()[CLIP_LIST]
    model, stands = remodel(sarc.Sarc(yaz0.decompress(archive)).files()[MODEL_FILE], char, clip_list)
    logger.info("  compressing %s (this is the slow part)", ARCHIVE)
    mod.write(ARCHIVE, repack(archive, {MODEL_FILE: model}))
    new_list, first = define_moves(define_stands(clip_list, stands), char.moves)
    mod.write(SETTINGS, repack(settings, {CLIP_LIST: new_list}))
    logger.info("  his clips are %#x to %#x of the game's list", first, first + len(char.moves) - 1)
    ptcl = sarc.Sarc(yaz0.decompress(effects)).files()[EFFECT_FILE]
    mod.write(EFFECTS, repack(effects, {EFFECT_FILE: hide_lights(ptcl)}))
    return Built(read_entries(new_list), first, char.moves)
