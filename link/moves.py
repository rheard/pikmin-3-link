"""
The code this mod adds to the game: Link's sword and spin, his roll, his bombs, what he does when he
is left standing, and his bow.

Everything else the mod does is data the game already reads. A new move is new behaviour, so this
part is code: an IPS patch of the game's program, which a loader applies before the game starts, and
a program file of the mod's own (`exefs/subsdk1`), which the loader lays out after the game's own.
The code itself is the assembly in `asm/`; this module gives it its numbers, assembles it and writes
both files.

What the code does, for the one leader whose model is Link's and for nobody else
--------------------------------------------------------------------------------
* **Swings** (`asm/swing.s`). Where a leader punches, he swings the Master Sword: the swing across
  from his right if he was standing, the swing straight down if he was moving, which is his own
  game's rule. Once a swing has passed its "the next press is taken from here" marker, a press of A
  plays the next swing at once, the kind the stick asks for. Each frame he is told to stop, or, for
  the frames a swing steps forward in his own game, to go the way he faces. The game's own code still
  does the hitting, on each clip's action marker, with his numbers in place of a leader's: harder,
  and reaching twice as far. With nobody following him, A swings at nothing too.
* **His roll** (the end of `asm/swing.s`). Where the game's dodge plays the leaders' roll forward,
  his own. Everything else of the dodge is the game's.
* **Bombs** (`asm/carry.s`, `asm/bombs.s`, `asm/put.s`). D-pad Down, which the game maps to
  nothing, puts him in the punch state with his legs still walking: he raises his hands, a bomb of
  the game's second kind is made over his head and hung on him, and he walks about holding it. A
  throws it; D-pad Down again sets it down. Either way it is lit as fire lights a bomb rock. A map
  has no pool of that kind of bomb outside Bingo Battle, so one is set up as a map's pools are.
* **His legs' frames** (`asm/put.s`). Where the game hands a part of a leader the frame its old
  clip was on as his speed changes, the frame is brought round into the new clip's loop first: his
  stand is far longer than the leaders', and his legs would otherwise stop on a walk's last frame.
* **Left standing** (`asm/idle.s`). Where the game would have a leader nobody plays play one of
  the Pikmin's clips once, he plays the next of five things of his own, in turn; where it sits him
  down he sits down on the ground, and he is not laid down to sleep.
* **His bow** (`asm/bow.s`). A click of the right stick takes it out: he raises it and walks about
  aiming. A looses an arrow, which flies on ahead of him; on each frame of its flight the punch
  state's own search looks for something for it to strike, where its point is. A click again puts
  the bow away. A blow that ends his aiming does not: he raises it again when he can.
* **His spin** (`asm/spin.s`). A held through the first frames of a swing takes him into his own
  game's charge; let go once it has charged he spins, and the search, turned round him to where the
  blade is on each frame, strikes everything round him once.

Where it lives
--------------
Twenty-two words of the game are changed (`asm/hooks.s`): nineteen call the new code, and three are
the number of Alph's sitting clip. Most of the code sits in stretches of the program that can never
run (`ROOM`): this is a Wii U port, and a block that served the GamePad is still there behind a
check that is now always false; so are the functions of two kinds of mission rule that no mission in
the game uses. The bow's and the spin's code is in the mod's own program file. A few bytes at the end
of the program's zeroed memory hold its state.

The patch fits one build of the game (`BUILD_ID`) and is named after it, so a loader will not apply
it to any other.
"""

from __future__ import annotations

import logging
import math
import struct

from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING

from link import arm
from link.character import ARROW_FRAMES, ARROW_SPEED, RATE, TURNS, WALK, bomb_put_path, bomb_seat
from link.errors import BuildError
from link.formats import nso

if TYPE_CHECKING:
    from link.character import Character
    from link.model import Built
    from link.oot.link import Link
    from link.romfs import Mod

logger = logging.getLogger(__name__)

BUILD_ID = "D467F5AD367BBEE88C51F8C468143CF23486A44B"  # the program of Pikmin 3 Deluxe 1.2.0
BASE = 0x60000000  # what the addresses here count from; a patch counts from 0x100 before the code's start
ASM = Path(__file__).parent / "asm"
SOURCES = ("swing", "carry", "bombs", "put", "idle", "bow", "spin")  # assembled as one text, in this order
LINK = 8  # what a leader's actor holds at +0x81c when its model is Alph's, which is now Link's

# The game's own functions and data that the code calls or reads, by address.
GAME = {
    "anim_flags": 0x60E2B77C,  # (animation): the markers it passed this frame; bit 31 at its end
    "player_of": 0x60BEDB3C,  # (leader): 0 or 1, the player controlling it, or -1
    "shared_mode": 0x60B12218,  # whether pads are assigned another way just now ...
    "shared_pad": 0x60E0A874,  # ... and if so, whose pad it is
    "mode_object": 0x6189C330,
    "game_object": 0x61EA24BC,
    "pads": 0x6189DE18,  # player n's pad is 0x10 + n * 0x4f0 into this
    "party_of": 0x60BF19C8,  # (leader): the party the game keeps for him (he is in it), or 0; how many at +0x68
    "facing_of": 0x6095F324,  # (leader): what holds the way he faces, at +0x3c (across) and +0x5c (along)
    "steer": 0x60C4E3B8,  # (state, leader, 0): the walk state's stick-to-movement, called once a frame
    "find_manager": 0x6099F448,  # (managers, name): the manager of one kind of thing, by the kind's name, or 0
    "managers": 0x61897E94,  # the game's list of managers
    "string_table": 0x61585038,  # the table of functions of the game's string holder, which a name is handed over in
    "mine_name": 0x6142F950,  # "SearchBomb": the game's own name for its second kind of bomb
    "birth_table": 0x615AD3C8,  # the table of functions of the record a new thing is made from
    "scope_begin": 0x6001D68C,  # (scope, name, memory): a stretch of memory of its own for what follows
    "scope_end": 0x6001D6EC,  # (scope): that stretch closed, cut down to what was used
    "manager_for": 0x603337F0,  # (managers, name, 1): the manager of a kind, made now if there is none
    "make_pool": 0x6099D7A4,  # (manager, memory, memory, how many): its pool
    "hold": 0x60A2D1A8,  # (slot, leader): take the leader's lock
    "release": 0x60A2D1C4,  # (slot): give it back
    "drop_pending": 0x609B9A78,  # (state machine): discard a state waiting to be entered
    "make_punch": 0x60C7C2F0,  # (state machine, 0x30): the punch state with no enemy named
    "walk_return": 0x60C50358,  # where the walk state goes after its own changes of state
    "layer": 0x60C8BBAC,  # (animation, 0 upper body or 1 legs, on): whether that part of him follows his speed
    "hang": 0x60D1C440,  # (a thing's link, how, the holder, where on it, what to follow): hang the thing on the holder
    "unhang": 0x60D1C208,  # (a thing's link): let it go
    "make_once": 0x609BD354,  # (behaviour, 1, what is asked for): start "play this clip once"
    "rest_waits": 0x609BCC00,  # where a rest sets its wait again and goes on
    "lock_on": 0x60C4DDFC,  # (state): the lock-on button taken, as the walk state takes it every frame
}
# Where the new code goes: the first address and the size in words of each stretch that cannot run.
ROOM = {
    "ROOM_SWORD": (0x60E422A8, 342),  # the functions of mission rule types 1 and 2, which no mission uses
    "ROOM_BOMBS": (0x60E375E4, 236),  # more of those functions
    "ROOM_PUT": (0x60E42198, 66),  # and more
    "ROOM_IDLE": (0x60C5ED70, 35),  # a block that served the Wii U GamePad
}
# The mod's own program file, laid out after the game's `main` and `subsdk0` (each to a whole page in
# memory), and its code past the module's header.
PROGRAM = BASE + 0x1EB8000 + 0x60F000
PROGRAM_CODE = PROGRAM + 0x100
PROGRAM_FILE = "subsdk1"  # its name in the mod's exefs folder: the first the game does not have
PROGRAM_NAME = "Link"  # the module's name, which the emulator prints as it loads it

# The code's memory, past the program's last variable: which swing of a run, "a press is being
# taken", which kind of swing, frames into it; frames until a bomb let go is lit, a bomb asked for,
# what he is doing with one, frames into setting it down, that bomb and which of its lives; which of
# the things he does left standing came last; then his bow's bytes and his spin's.
STATE = 0x61EB7F00
BYTES = {
    "COUNT": 0, "TAKING": 1, "KIND": 2, "FRAMES": 3, "PEND": 4, "ASKED": 5, "MODE": 6, "PLACE": 7, "BOMB": 8,
    "SERIAL": 12, "IDLE": 16, "PATCHED": 0x20, "BOW_ASKED": 0x21, "BOW_MODE": 0x22, "BOW_FLIGHT": 0x23,
    "BOW_HIT": 0x24, "BOW_KEPT": 0x25, "SPIN_MODE": 0x28, "SPIN_FRAMES": 0x29, "SPIN_HELD": 0x2A,
    "SPIN_COUNT": 0x2B, "SPIN_LIST": 0x2C,
}  # fmt: skip
# What he is doing with a bomb, with his bow, and in a charge.
MODES = {
    "RAISING": 1, "HOLDING": 2, "THROWING": 3, "PUTTING": 4,
    "BOW_RAISING": 1, "BOW_AIMING": 2, "BOW_LOOSING": 3, "BOW_TAKING": 4,
    "CHARGING": 1, "SPINNING": 2, "LOWERING": 3,
}  # fmt: skip
# The game's own clips the code names, by their place in its list.
THEIR_ROLL = 0x5A  # the leaders' roll forward, which the game's dodge plays
THEIR_STAND = 0x1C  # the stand every Pikmin shares

# A leader's punch: a sphere of `radius` centred `ahead` of him is searched for enemies, and the
# nearest within `furthest` of him is hit with `power`.
THEIRS = {"ahead": 4.0, "radius": 10.0, "furthest": 20.0, "power": 3.0}
# His swing: a blow of 30 costs a Dwarf Bulborb 60% of its life (a blow goes to both of its damage
# nodes), and it reaches twice as far as a leader's punch.
POWER, REACH = 30.0, 2.0
PUSHED = 0.3  # how far over the stick must be to count as pushed: where the walk state begins to turn him
# His spin: twice a swing's blow, struck once on each thing round him, and how many it can keep.
SPIN_POWER = 60.0
SPIN_HITS = 24
# His bomb: made where he holds it, lit two frames after it is let go (its own first frame clears
# the byte that lights one), going off 3 seconds after that, and thrown a little faster than first
# set (`FARTHER`), both ways alike. A map is given a pool of 8.
BOMB_SPEED = (130.0, 110.0)  # units a second, forward and upward
FARTHER = 1.15
GRAVITY = 400.0  # this game's, in units a second each second
BOMB_WAIT = 2
BOMB_FUSE = 3.0
BOMB_POOL = 8
# His arrow: a blow of a swing's strength, searched for in a sphere this wide, this far above the
# level of his feet; and while it flies, no limit on how far away what it strikes may be.
ARROW_POWER = 30.0
ARROW_RADIUS = 12.0
ARROW_RISE = 8.0
ARROW_FAR = 1000.0


def bits(value: float) -> int:
    """A number as the 32 bits the game keeps it in."""
    return struct.unpack("<I", struct.pack("<f", value))[0]


def his_hit(power: float) -> dict[str, float]:
    """
    What a blow of his sword reaches, and how hard it lands. The sphere searched grows forward and to
    the sides; its back edge stays where a leader's is, so that he does not start hitting what is
    behind him.

    Returns:
        The same four numbers as `THEIRS`.
    """
    radius = THEIRS["radius"] * REACH
    return {
        "power": power,
        "radius": radius,
        "ahead": radius - (THEIRS["radius"] - THEIRS["ahead"]),
        "furthest": THEIRS["furthest"] * REACH,
    }


def flight(forward: float, upward: float, above: float) -> float:
    """How far ahead a bomb let go `above` the ground at these speeds comes down, by the game's gravity alone."""
    rise = upward**2 / (2.0 * GRAVITY) if upward > 0 else 0.0
    up = max(upward, 0.0) / GRAVITY
    down = (2.0 * (above + rise) / GRAVITY) ** 0.5
    return forward * (up + down)


def farther(ahead: float, above: float) -> tuple[float, float]:
    """
    The throw's speeds, raised alike (so that it leaves at the same angle) until it comes down
    `FARTHER` times as far ahead of him; each to a tenth of a unit a second.

    Returns:
        The speeds forward and upward.
    """
    forward, upward = BOMB_SPEED
    want = FARTHER * (ahead + flight(forward, upward, above))
    low, high = 0.25, 4.0
    for _ in range(60):
        mid = (low + high) / 2.0
        if ahead + flight(forward * mid, upward * mid, above) < want:
            low = mid
        else:
            high = mid
    return round(forward * low, 1), round(upward * low, 1)


def symbols(char: Character, clips: Built, plain: Link, radius: float) -> tuple[dict[str, int], dict[str, list[int]]]:
    """
    Every name the assembly uses that is not one of its own labels, and the tables it holds.

    `plain` is Link with open hands, for where a bomb sits in them, and `radius` the bomb's.

    Returns:
        The symbols, and the tables by name.
    """
    numbers = clips.numbers
    hit, spin_hit = his_hit(POWER), his_hit(SPIN_POWER)
    moves = char.moves
    seat = bomb_seat(plain, char.scale, radius)
    path = bomb_put_path(plain, char.scale, radius, moves["l_bomb_put"].more)
    forward, upward = farther(*seat)
    (_, _, ahead), _ = char.bow_flight()
    tips = [ahead + char.arrow.length * char.scale + ARROW_SPEED * k for k in range(ARROW_FRAMES + 1)]
    reaches = [(a + b) / 2.0 for a, b in pairwise(tips)]
    spins = char.link.spins()
    sweep = char.spin_sweep()
    floats = {
        "PUSHED": PUSHED,
        "RADIUS_THEIRS": THEIRS["radius"],
        "FURTHEST_THEIRS": THEIRS["furthest"],
        "POWER_THEIRS": THEIRS["power"],
        "RADIUS": hit["radius"],
        "AHEAD": hit["ahead"],
        "FURTHEST": hit["furthest"],
        "POWER": hit["power"],
        "BOMB_FUSE": BOMB_FUSE,
        "BOMB_AHEAD": seat[0],
        "BOMB_ABOVE": seat[1],
        # (the game's "knocked away" takes a speed by the frame, and makes it one by the second itself)
        "BOMB_FORWARD": forward / 30.0,
        "BOMB_UPWARD": upward / 30.0,
        "ARROW_RISE": ARROW_RISE,
        "ARROW_RADIUS": ARROW_RADIUS,
        "ARROW_FAR": ARROW_FAR,
        "ARROW_POWER": ARROW_POWER,
        "SPIN_OUT": spin_hit["ahead"],
        "SPIN_RADIUS": spin_hit["radius"],
        "SPIN_POWER": SPIN_POWER,
    }
    names = {
        **GAME,
        **{name: start for name, (start, _) in ROOM.items()},
        **BYTES,
        **MODES,
        **{name: bits(value) for name, value in floats.items()},
        "PROGRAM_CODE": PROGRAM_CODE,
        "STATE": STATE,
        "LINK": LINK,
        "SWING_CLIP": numbers["l_slash1"],
        "RUN": 3,
        "THEIR_ROLL": THEIR_ROLL,
        "ROLL_CLIP": numbers["l_roll"],
        "LIFT_CLIP": numbers["l_bomb_up"],
        "HOLD_CLIP": numbers["l_bomb_hold"],
        "THROW_CLIP": numbers["l_bomb_throw"],
        "PUT_CLIP": numbers["l_bomb_put"],
        "PUT_FRAMES": len(path),
        "BOMB_WAIT": BOMB_WAIT,
        "BOMB_POOL": BOMB_POOL,
        "IDLE_CLIP": numbers["l_glance"],
        "TURNS": TURNS,
        "SIT_CLIP": numbers["l_sit"],
        "THEIR_STAND": THEIR_STAND,
        "HIS_STAND": clips.stand,
        "READY_CLIP": numbers["l_bow_ready"],
        "AIM_CLIP": numbers["l_bow_aim"],
        "LOOSE_CLIP": numbers["l_bow_loose"],
        "NEXT_CLIP": numbers["l_bow_next"],
        "FLIGHT": len(reaches),
        "EXPECT": 0,
        "CHARGE_CLIP": numbers["l_charge"],
        "SPIN_CLIP": numbers["l_spin"],
        "LOWER_CLIP": numbers["l_lower"],
        "HANDOVER": math.ceil(spins["handover"] * RATE),
        "SPIN_READY": math.ceil((char.spin_keys("charge").lead + spins["ready"]) * RATE),
        "SPIN_UNTIL": len(sweep) - 1,
        "SPIN_LAST": len(sweep) - 1,
        "SPIN_HITS": SPIN_HITS,
    }
    steps = [move.step for move in moves.values() if move.family == "swing"]
    tables = {
        "SWING_STEPS": [n for frames, speed in steps for n in (frames, bits(speed / WALK if frames else 0.0))],
        "PUT_PATH": [n for ahead_, up in path for n in (bits(up), bits(ahead_))],
        "ARROW_REACHES": [bits(r) for r in reaches],
        "SPIN_SWEEP": [n for a, _ in sweep for n in (bits(math.cos(a)), bits(math.sin(a)))],
    }
    return names, tables


def program(names: dict[str, int], tables: dict[str, list[int]]) -> tuple[dict[int, int], dict[int, int]]:
    """
    Assemble the code.

    The bow's code first looks for a word of the patch's own code where the patch puts it (the two
    files are separate, and one may be loaded without the other), so it is assembled twice: once to
    know that word, and once with it.

    Returns:
        The words of the patch, and the words of the program file, by address.

    Raises:
        BuildError: If the code does not fit the room set aside for it.
    """
    text = "\n".join((ASM / f"{source}.s").read_text(encoding="utf-8") for source in SOURCES)
    try:
        words, labels = arm.assemble(text, names, tables)
        names = {**names, "EXPECT": words[labels["b_walk"] + 4]}
        words, labels = arm.assemble(text, names, tables)
        hooks, _ = arm.assemble((ASM / "hooks.s").read_text(encoding="utf-8"), {**names, **labels})
    except arm.AsmError as error:
        raise BuildError(f"asm: {error}") from error
    own = {address: word for address, word in words.items() if address >= PROGRAM_CODE}
    patch = {address: word for address, word in words.items() if address < PROGRAM_CODE}
    for start, size in ROOM.values():
        inside = sorted(a for a in patch if start <= a < start + 4 * size)
        if inside != list(range(start, start + 4 * len(inside), 4)) or len(inside) > size:
            raise BuildError(f"the code leaves a hole in the room at {start:#x}, or overflows it")
    if len(patch) != sum(any(start <= a < start + 4 * size for start, size in ROOM.values()) for a in patch):
        raise BuildError("some of the code is outside the room set aside for it")
    if sorted(own) != list(range(PROGRAM_CODE, PROGRAM_CODE + 4 * len(own), 4)):
        raise BuildError("the program file's code is not one stretch")
    if patch.keys() & hooks.keys():
        raise BuildError("a changed word of the game is where the code is")
    return {**patch, **hooks}, own


def ips(words: dict[int, int]) -> bytes:
    """
    An IPS patch of the program holding `words`.

    Each run of words is one record at its offset in the program file as a loader sees it, which is
    0x100 past the start of the code.

    Returns:
        The patch file's bytes.

    Raises:
        BuildError: If a run cannot be held in the format.
    """
    patch = bytearray(b"PATCH")
    run: list[int] = []
    for address in [*sorted(words), None]:
        if run and address != run[-1] + 4:
            offset = run[0] - BASE + 0x100
            data = b"".join(struct.pack("<I", words[a]) for a in run)
            if offset >= 0x1000000 or offset == int.from_bytes(b"EOF", "big") or len(data) > 0xFFFF:
                raise BuildError(f"the run of code at {run[0]:#x} does not fit the patch format")
            patch += offset.to_bytes(3, "big") + len(data).to_bytes(2, "big") + data
            run = []
        if address is not None:
            run.append(address)
    return bytes(patch + b"EOF")


def build(char: Character, clips: Built, plain: Link, radius: float, mod: Mod) -> None:
    """
    Write the patch and the program file into the mod folder.

    `clips` is where his clips are in the game's list, `plain` Link with open hands, and `radius` the
    radius of his bomb, as the bomb's files have it.
    """
    names, tables = symbols(char, clips, plain, radius)
    patch, own = program(names, tables)
    code = b"".join(struct.pack("<I", own[address]) for address in sorted(own))
    mod.write_patch(BUILD_ID, ips(patch))
    mod.write_program(PROGRAM_FILE, nso.build(PROGRAM_NAME, code, PROGRAM_CODE - PROGRAM))
    logger.info(
        "  %d words of the game changed, %d words of code in its unused stretches and %d in the mod's own program file",
        len(patch) - sum(any(s <= a < s + 4 * n for s, n in ROOM.values()) for a in patch),
        sum(any(s <= a < s + 4 * n for s, n in ROOM.values()) for a in patch),
        len(own),
    )
