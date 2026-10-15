"""
Link's voice: what he says when he throws a Pikmin, swings his sword and spins, and wherever the
game has Alph speak.

What is written
---------------
* `audio/audiores/CarrotAudio.bfsar`: the game's sound archive, with entries for his sounds, and
  Alph's own voice sent to them.
* `audio/audiores/GROUP_STATIC.bfgrp`: the one file all the game's sounds are loaded from, with his
  on the end.
* `command/command_table.szs`: the table that ties sounds to frames of animation clips.

How, all of it data the game already reads
------------------------------------------
* **When.** The game ties sounds to frames of animation clips in a text table. The leaders' punch
  is the pattern: one record plays a grunt at frame 0 of the punch clip. A copy of that record does
  the same for the throw, and one each for the clips of his swings and of his spin.
* **Who.** The punch's grunt is a family of sounds, one run for each kind of Pikmin and each leader,
  and the game picks the run by who is making the sound. Each new family has Link in Alph's run and
  a moment of silence at no volume in all the others, so nobody else changes.
* **Which.** What he says is read from the code that moves him, as numbers of his voice. His own
  game does not play one recording for a number: its script picks one of several each time
  (`oot.audio`). A sound here can say "pick among the next N at random, with these weights", which
  is how each leader already has two punch grunts, so his run has every recording of the number,
  weighted as his game picks them. The game's own spacing rule keeps the throw from shouting on
  every Pikmin of a fast volley.
* **What.** The sounds themselves are new: a wave archive and a wave sound file added to the end of
  the group file, and entries for them added to the sound archive. None of that touches a stock
  sound. His recordings go in as his ROM keeps them, at the rate his game plays them at (20 kHz),
  not resampled.
* **How loud.** His game plays his voice at about two thirds of full loudness against its own
  sounds. Here his recordings are at the level they are stored at, and they cannot be made louder
  than that without clipping, so how loud he is is set by the volume of the sound that plays each:
  `VOLUME`, a third louder than the leaders' punch grunt, at whose 96 he is too quiet beside them.
* **In place of Alph's.** The rest of a leader's voice has no clip of his own to hang a record on:
  the game plays it by name from its code, or from records on clips the Pikmin share, and Alph's
  run of each family is recordings of Alph. Those sounds are the one thing of the game's own that
  is changed here: each is sent to one of Link's recordings (`SWAPS`, `MORE`), and keeps everything
  else about how it is played but its loudness. Only recordings of Alph are changed, and none that
  another leader is heard to make, so nobody else's voice is.
"""

from __future__ import annotations

import logging
import math
import re
import struct

from functools import cache, partial
from itertools import starmap
from typing import TYPE_CHECKING, NamedTuple

from link.character import SPINS, SWINGS
from link.errors import BuildError
from link.formats import bfsar, sarc, yaz0
from link.oot.audio import Voice
from link.romfs import repack

if TYPE_CHECKING:
    from collections.abc import Callable

    from link.oot.link import Link
    from link.romfs import Mod, RomFs

logger = logging.getLogger(__name__)

ARCHIVE = "audio/audiores/CarrotAudio.bfsar"
GROUP = "audio/audiores/GROUP_STATIC.bfgrp"
COMMANDS = "command/command_table.szs"
TABLE = "sound_table.txt"

LEADER = "A"  # Alph
PATTERN = "Pikmin/punch/Attack/0"  # the leaders' punch grunt: the record and the family that are copied
KIND = 2  # the pattern's way of choosing a run by who makes the sound: `bfsar.TYPE_SLOTS`
SILENCE = 512  # samples of nothing for everyone else: 16 ms
SILENCE_RATE = 32000
WAVE_ARCHIVE_ITEM = 0x05000000  # an item number's top byte says what kind of thing it names
SOUND_ITEM = 0x01000000
MAIN_GROUP = [0x06000000]

# How loud his lines are, and the first of Alph's sounds he takes over (`SWAPS`). The setting is a
# byte; the game's own leader voice lines go up to 140 and its sounds up to 250.
VOLUME = 128
# The throw: its clip and the frame of it where the clip's own throw marker is, and not again for 7
# to 15 frames. His swings and his spin are shouted on frame 0 of their clips, every time.
THROW = ("throw", 1)
THROW_GAP, THROW_GAP_SPREAD = 7, 8
LOUDEST = 0.1  # seconds: a recording is as loud as its strongest stretch this long (`strongest`)

# What a leader says that the game plays by name from its code (or from a record of its own on a clip
# the leaders share with the Pikmin), given to Link at `VOLUME`: (when it is heard, the number of his
# voice, the archive's names for the sounds). The game picks among the sounds of one event, and his
# recordings are given out to them in turn.
#
# Which of his: by the names the public documentation of his game's source gives the numbers; nothing
# in his ROM names them. 0x6814 is what he says as he jumps off an edge by himself; 0x6816 what he
# says when he is surprised; 0x6805 when he is hurt (three recordings, given out in turn to Alph's
# seven); 0x6807 a short fall (his code asks for it, and for 0x6808, the long one, in one routine
# that also starts his roll, which fits a fall and its landing); 0x6804 a short effort. For the
# dodge, None: the one number his own roll asks his voice for, read from his code
# (`Link.roll_voice`).
SWAPS = (
    # (every time; the second copy of it, which no record of the sound table plays, is in `MORE`)
    ("when the player changes to him", 0x6814, ("WSD_Player{}Vo_Change",)),
    # (also as the player comes back to him resting; it is said beside the hop of the clip `kizuku`,
    # whose own sound, the same for all five leaders, is left as it is)
    ("as he notices something and joins a party", 0x6816, ("WSD_PikminVo_JoinPartyPlayer{}",)),
    (
        "when he is hit, caught by a blast or squashed",
        0x6805,
        (
            "WSD_PikminVo_DamagePlayer{}",
            "WSD_PikminVo_DamagePlayer{}2",
            "WSD_PikminVo_BombDamagePlayer{}",
            "WSD_PikminVo_BombDamagePlayer{}2",
            "WSD_Player{}Vo_Squash",
            "WSD_Player{}Vo_Squash2",
            "WSD_Player{}Vo_Squash3",
        ),
    ),
    (
        "when another leader throws him",
        0x6807,
        ("WSD_PikminVo_ThrownPlayer{}", "WSD_PikminVo_ThrownPlayer{}2", "WSD_PikminVo_ThrownPlayer{}3"),
    ),
    (
        "when he is dismissed from a party",
        0x6804,
        ("WSD_PikminVo_BeFreePlayer{}", "WSD_PikminVo_BeFreePlayer{}2", "WSD_PikminVo_BeFreePlayer{}3"),
    ),
    # (the leader being played makes the roll's own sound and says nothing)
    (
        "when a party he is in dodges",
        None,
        ("WSD_PikminVo_StepOutPlayer{}", "WSD_PikminVo_StepOut2Player{}", "WSD_PikminVo_StepOut3Player{}"),
    ),
)
# The rest of the leader's own voice: (when it is heard, ((the number of his voice, the archive's names
# for the sounds the game picks among for one event), ...)). These events the game sets anywhere from
# volume 30 (pumping his fist) to 100 (a trampoline), so each is made as loud as Alph's was
# (`matched`). Measured that way, the sounds of `SWAPS` at `VOLUME` come out about as loud as Alph's.
#
# Which of his, by the documentation's names again. He has few sounds and no glad ones, so each is the
# nearest thing he says in his own game:
#   0x6814  what he says as he jumps off an edge: wherever a leader says something as he jumps
#   0x6801  the longer shout of the later blow of a run of swings, his loudest effort: setting off
#           down a slide, sent up by a trampoline, and the jump of the ultra-spicy spray
#   0x6800  his shout as he swings or throws, which is also what he says as he hops to a side or flips
#           back: carried up by a geyser, and (quietly, as the game has Alph's) as he pumps his fist
#   0x6808  his long cry as he falls a long way: into a pit
#   0x680D  held by something that has caught him: in a net. It plays the same three recordings as
#           0x6805, which go in once
#   0x6805  hurt: knocked back, and running about in a panic
#   0x6816  surprised: as a bubble he is caught in bursts
#   0x6804  the effort of pulling himself up onto a ledge: getting up after being knocked down
#
# Left as they are, of the recordings the archive names for Alph:
#   * what he says as he waves: his game has no greeting, or any word, to put there;
#   * the leaders' yawn, which he no longer plays (what he does left standing is his own), and whose
#     match in his game (0x6811) is three notes in a row, which `oot.audio` does not read;
#   * the cutscenes', which play beside a model of Alph that this mod does not replace;
#   * the search of a leader left standing, and the punch grunt, which are not his alone: in families
#     made for the Pikmin alone, other leaders' places fall on them (see `bfsar.TYPE_SLOTS`);
#   * an alarm, which is not a voice; and the ones that are silent in the game's own files, or that
#     nothing plays for him.
# Four of those below can be reached the same way by another leader, through families made for
# things only a Pikmin does (climbing a stalk, throwing what it carries), and are changed all the
# same: the three of the clip `drop`, and the third of the pit.
MORE = (
    (
        "in the second copy of what he says when the player changes to him",
        ((0x6814, ("WSD_Player{}Vo_Change_Multi",)),),
    ),
    (
        "as he jumps: the clips `drop_lotus`, `jump` and `drop`",
        (
            (
                0x6814,
                (
                    "WSD_PikminVo_JumpToIndianLotusPlayer{}",
                    "WSD_PikminVo_JumpToIndianLotusPlayer{}2",
                    "WSD_PikminVo_JumpToIndianLotusPlayer{}3",
                ),
            ),
            (0x6814, ("WSD_PikminVo_JumpPlayer{}1", "WSD_PikminVo_JumpPlayer{}2", "WSD_PikminVo_JumpPlayer{}3")),
            (0x6814, ("WSD_Player{}Vo_Drop", "WSD_Player{}Vo_Drop2", "WSD_Player{}Vo_Drop3")),
        ),
    ),
    (
        "as he sets off down a slide",
        ((0x6801, ("WSD_Player{}Vo_SliderStart", "WSD_Player{}Vo_SliderStart2", "WSD_Player{}Vo_SliderStart3")),),
    ),
    (
        "on a trampoline, and carried up by a geyser",
        (
            (0x6801, ("WSD_Player{}Vo_Trampoline", "WSD_Player{}Vo_Trampoline2")),
            (0x6800, ("WSD_Player{}Vo_RideOnGeyser", "WSD_Player{}Vo_RideOnGeyser2")),
        ),
    ),
    (
        "as he jumps for joy, and with the ultra-spicy spray",
        (
            (0x6814, ("WSD_PikminVo_GladJumpPlayer{}", "WSD_PikminVo_GladJumpPlayer{}2")),
            (0x6801, ("WSD_PikminVo_GekikaraJumpPlayer{}",)),
        ),
    ),
    (
        "as he pumps his fist",
        ((0x6800, ("WSD_Player{}Vo_Gattsu", "WSD_Player{}Vo_Gattsu2", "WSD_Player{}Vo_Gattsu3")),),
    ),
    (
        "as he falls into a pit",
        (
            (
                0x6808,
                (
                    "WSD_PikminVo_FallIntoHellPlayer{}",
                    "WSD_PikminVo_FallIntoHellPlayer{}2",
                    "WSD_PikminVo_FallIntoHellPlayer{}3",
                ),
            ),
        ),
    ),
    ("caught in a net", ((0x680D, ("WSD_Player{}Vo_Net", "WSD_Player{}Vo_Net2", "WSD_Player{}Vo_Net3")),)),
    # (the second of the first three is spelled so in the game's archive)
    (
        "knocked back, and as a bubble he is caught in bursts",
        (
            (
                0x6805,
                ("WSD_PikminVo_ReboundPlayer{}", "WSD_PikminrVo_ReboundPlaye{}2", "WSD_PikminVo_ReboundPlayer{}3"),
            ),
            (0x6816, ("WSD_PikminVo_BubbleBreakPlayer{}", "WSD_PikminVo_BubbleBreakPlayer{}2")),
        ),
    ),
    (
        "as he gets up after being knocked down",
        (
            (
                0x6804,
                (
                    "WSD_PikminVo_GetUpFromElectricShockPlayer{}",
                    "WSD_PikminVo_GetUpFromElectricShockPlayer{}2",
                    "WSD_PikminVo_GetUpFromElectricShockPlayer{}3",
                ),
            ),
        ),
    ),
    # (the first of the game's three for this is silent, and the one it picks most often: left as it is)
    ("running about in a panic", ((0x6805, ("WSD_PikminVo_RunAwayPlayer{}2", "WSD_PikminVo_RunAwayPlayer{}3")),)),
)


class Take(NamedTuple):
    """One recording he may say: its name, its weight among a line's takes, its samples and their rate."""

    name: str
    weight: int
    samples: list[int]
    rate: int


class Line:
    """
    One thing he says: at `frame` of the clip tagged `clip`, one of `takes`, picked by weight.

    `gap` and `gap_spread` keep it from being said again for that many frames, and up to that many
    more at random.
    """

    def __init__(self, clip: str, frame: int, takes: list[Take], gap: int = 0, gap_spread: int = 0) -> None:
        self.clip, self.frame, self.takes, self.gap, self.gap_spread = clip, frame, list(takes), gap, gap_spread
        self.name = f"Pikmin/{clip}/LeaderVoice/{frame}"


class Swap(NamedTuple):
    """One of the game's own sounds, by the archive's name for it, that is to play a take of his, and how loud."""

    label: str
    take: Take
    volume: int


def shouts(voice: Voice, number: int) -> list[Take]:
    """
    What a number of his voice plays: a take for each recording his game picks among for it.

    Each is named by the number and its place, weighted by its share of the times, and written as
    his ROM keeps it: not turned down as his game plays it, since the volume of the sound that plays
    it sets how loud he is.

    Returns:
        The takes.

    Raises:
        BuildError: If the shares are not weights of 1 to 15, or there are more recordings than the
            game picks among.
    """
    by_sample = {}  # (two takes of one recording, at two loudnesses, are one sound here)
    for sound in voice.play(number):
        entry = by_sample.setdefault(sound.sample.address, [sound, 0])
        entry[1] += sound.share
    scale = math.lcm(*(share.denominator for _, share in by_sample.values()))
    out = []
    for k, (sound, share) in enumerate(by_sample.values()):
        weight = int(share * scale)
        if not 1 <= weight <= 15:
            raise BuildError(f"a recording of {number:#06x} is played {share} of the times: not a weight of 1 to 15")
        samples = voice.decoded(sound.sample)
        out.append(Take(f"{number:04x}_{k + 1}", weight, samples, sound.rate))
        logger.info(
            "  %-7s sound effect %d of font %d: %.2f s at %d Hz, %s of the times",
            out[-1].name,
            sound.effect,
            sound.font,
            len(samples) / float(sound.rate),
            sound.rate,
            share,
        )
    if not 1 <= len(out) <= 7:
        raise BuildError(f"{number:#06x} plays one of {len(out)} recordings; the game picks among at most 7")
    return out


def lines(voice: Voice, recordings: Callable[[int], list[Take]], link: Link) -> list[Line]:
    """
    What he says on clips: as he throws, as each swing of his sword starts and as his spin starts.

    Each number of his voice is read from the code that moves him (`oot.audio.Voice`). As he
    throws: the voice his code asks for beside the throw's swish. As he swings: his game chooses
    in one routine, one number for a swing (on the first release the throw's) and another for the
    later blow of a run; every swing shouts, as there, with no frames kept clear between two shouts.
    As he spins: the same routine asks, for a record of the table of swings that is a spin, for one
    number and no swish (on the first release the later blow's). The blade's own swish, which his
    game plays beside the shout, is an instrument of another bank, and is not read.

    `recordings` gives the takes of a number of his voice, as `shouts` does.

    Returns:
        A line for the throw, then one for each of his swings (`SWINGS`), then one for his spin.

    Raises:
        BuildError: If the records of the table of swings the routine takes as spins are not where
            his spins are.
    """
    file = link.move_table()[0]
    swish, number = voice.thrown(file)
    logger.info("  as he throws, his code asks for sounds %#06x and %#06x: the swish and his voice", swish, number)
    out = [Line(*THROW, recordings(number), THROW_GAP, THROW_GAP_SPREAD)]
    first, later, _ = voice.swung(file)
    logger.info("  as he swings his sword, his voice %#06x, and %#06x with the later blow of a run", first, later)
    out += [Line(clip, 0, recordings(later if kind.endswith("_combo") else first)) for clip, kind in SWINGS]
    least, spun = voice.spun(file)
    records = link.spins()["records"]
    if least != min(records):
        raise BuildError(f"his code gives the spin's shout from record {least} on, and his spins are records {records}")
    logger.info("  as he spins, his voice %#06x", spun)
    out.append(Line(SPINS[1][0], 0, recordings(spun)))
    return out


def several(recordings: Callable[[int], list[Take]], number: int, count: int) -> list[Take]:
    """The takes of a number of his voice for `count` sounds of the game's: given out in turn, and round again."""
    takes = recordings(number)
    return [takes[k % len(takes)] for k in range(count)]


def strongest(samples: list[int], rate: int) -> float:
    """
    How strong a recording is where it is strongest: the root mean square of its strongest
    `LOUDEST` seconds (of all of it, if it is shorter).

    Returns:
        The root mean square, of 16-bit samples.
    """
    n = max(1, min(len(samples), int(rate * LOUDEST)))
    run = best = sum(v * v for v in samples[:n])
    for k in range(n, len(samples)):
        run += samples[k] * samples[k] - samples[k - n] * samples[k - n]
        best = max(best, run)
    return math.sqrt(best / float(n))


def matched(theirs: list[tuple[int, list[int], int]], his: list[Take]) -> list[int]:
    """
    The volume for each of his takes that makes an event as loud as the leader's own was.

    `theirs` is the game's sounds for the event, as (volume, samples, rate). The event's loudness
    is taken as the volume times how strong the recording is (`strongest`), averaged over the
    sounds the game picks among.

    Returns:
        A volume for each of `his`.

    Raises:
        BuildError: If a sound is silent, or a volume would be outside 1 to 255.
    """
    was = [volume * strongest(samples, rate) for volume, samples, rate in theirs]
    if not was or not all(was):
        raise BuildError("one of the game's sounds for an event is silent: there is no loudness to match")
    aim = sum(was) / len(was)
    out = []
    for take in his:
        mine = strongest(take.samples, take.rate)
        if not mine:
            raise BuildError(f"{take.name} is silent")
        volume = round(aim / mine)
        if not 1 <= volume <= 255:
            raise BuildError(f"{take.name} would need volume {volume} to be as loud as the sound it replaces")
        out.append(volume)
    return out


def resolve(archive: bfsar.Archive, group: bfsar.Group, sound: int) -> dict:
    """
    Follow one of the archive's wave sounds to its samples, the way the sound library does: the
    sound's file in the group, the wave it names, that wave's archive, the wave file.

    Returns:
        The wave, as `bfsar.read_wave` reads it.

    Raises:
        BuildError: If it is not a wave sound, or it names something that is not a wave archive.
    """
    info = archive.sound(sound)
    if info["kind"] != bfsar.WAVE_SOUND:
        raise BuildError(f"sound {sound:#x} is not a wave sound")
    item, number, _ = bfsar.read_wave_sounds(group.file(info["file"]))[info["index"]]
    if item >> 24 != WAVE_ARCHIVE_ITEM >> 24:
        raise BuildError(f"sound {sound:#x} names item {item:#x}, which is not a wave archive")
    holder = archive.wave_archive(item & 0xFFFFFF)
    at = next(group.body + off for fid, off, _ in group.items if fid == holder)
    off, _ = bfsar.read_wave_archive(group.d, at)[number]
    return bfsar.read_wave(group.d, off)


def own_voice(recordings: Callable[[int], list[Take]], link: Link, archive: bytes, group: bytes) -> list[Swap]:
    """
    What he says in place of the leader's own voice: `SWAPS` at `VOLUME`, then `MORE`, each event
    as loud as Alph's was. A recording that two numbers of his play goes in once, under the name it
    came by first. `recordings` gives the takes of a number of his voice, as `shouts` does.

    Returns:
        The swaps, in that order.
    """
    kept: dict[tuple[int, tuple[int, ...]], Take] = {}

    def once(take: Take) -> Take:
        return kept.setdefault((take.rate, tuple(take.samples)), take)

    out = []
    for _, number, names in SWAPS:
        voiced = link.roll_voice() if number is None else number
        if number is None:
            logger.info("  as he rolls, his code asks his voice for %#06x", voiced)
        labels = [name.format(LEADER) for name in names]
        out += [
            Swap(label, once(take), VOLUME)
            for label, take in zip(labels, several(recordings, voiced, len(labels)), strict=True)
        ]
    stock, grp, named = bfsar.Archive(archive), bfsar.Group(group), bfsar.labels(archive)
    for _, parts in MORE:
        for number, names in parts:
            labels = [name.format(LEADER) for name in names]
            theirs = []
            for label in labels:
                sound = stock_sound(named, label)
                wave = resolve(stock, grp, sound)
                theirs.append((stock.sound(sound)["volume"], wave["samples"], wave["rate"]))
            given = [once(take) for take in several(recordings, number, len(labels))]
            out.extend(starmap(Swap, zip(labels, given, matched(theirs, given), strict=True)))
    return out


def picks(archive: bfsar.Archive, sound: int, slot: int) -> list[tuple[int, int]]:
    """
    What the game may play for a record naming `sound`, made by an actor in run `slot`.

    Returns:
        (sound number, weight) for each candidate.
    """
    first = archive.sound(sound)["user"]
    if bfsar.user_field(first, "kind"):
        sound += slot * max(1, bfsar.user_field(first, "variants"))
    count = bfsar.user_field(archive.sound(sound)["user"], "variants")
    if count < 2:
        return [(sound, 16)]
    return [(sound + k, bfsar.user_field(archive.sound(sound + k)["user"], "weight") or 16) for k in range(count)]


def table_parts(table: bytes) -> tuple[int, bytes, int, list[bytes]]:
    """
    Take the sound table apart.

    Returns:
        How many records it has, the pattern's record, the largest key id, and every record's name.

    Raises:
        BuildError: If the table is not laid out as expected.
    """
    match = re.match(rb"# table version\r\n\d+\r\n# record num\r\n(\d+)\r\n\{\r\n", table)
    if not match or not table.endswith(b"\t}\r\n}\r\n"):
        raise BuildError("the sound table does not begin and end as expected")
    names = re.findall(rb'# command name\r\n\t\t\t"([^"]*)"', table)
    if len(names) != int(match.group(1)):
        raise BuildError(f"the sound table says {int(match.group(1))} records and holds {len(names)}")
    at = table.index(b'"' + PATTERN.encode() + b'"')
    start = table.rindex(b"\t# (OBJ)", 0, at)
    end = table.index(b"\t# (OBJ)", at)
    keys = [int(k) for k in re.findall(rb"# key id\r\n\t\t\t(\d+)", table)]
    return int(match.group(1)), table[start:end], max(keys), names


def stock_sound(names: dict[str, int], name: str) -> int:
    """
    One of the game's own sounds, by the name the sound archive gives it.

    Returns:
        The sound's number.

    Raises:
        BuildError: If the archive names no sound so.
    """
    item = names.get(name, 0)
    if item >> 24 != SOUND_ITEM >> 24:
        raise BuildError(f"the sound archive has no sound named {name}")
    return item & 0xFFFFFF


def _swap_once(text: bytes, old: bytes, new: bytes) -> bytes:
    if text.count(old) != 1:
        raise BuildError(f"the pattern record does not hold {old!r} exactly once")
    return text.replace(old, new)


def _set_field(record: bytes, field: bytes, value: bytes, old: bytes = rb"\d+") -> bytes:
    """A record with the number on the line after `# <field>` replaced (or whatever there matches `old`)."""
    return re.sub(rb"(# " + field + rb"\r\n\t\t\t)" + old, lambda match: match.group(1) + value, record, count=1)


def add_voice(
    archive: bytes,
    group: bytes,
    table: bytes,
    said: list[Line],
    own: list[Swap],
) -> tuple[bytes, bytes, bytes]:
    """
    Add his voice to the game's three files: a family of sounds and a record for each line.

    `own` is the game's own sounds that are to play a take of his instead. Such a sound keeps its
    place and everything that says how it is played (on which player, how it is picked among its
    fellows), but for its volume and its recording.

    Returns:
        The new sound archive, group file and sound table.

    Raises:
        BuildError: If the game's files are not put together the way this expects.
    """
    arc, grp = bfsar.Archive(archive), bfsar.Group(group)
    count, record, top_key, names = table_parts(table)
    if not said or len({line.name for line in said}) != len(said):
        raise BuildError("each line needs a clip and frame of its own")

    # The pattern: the punch grunt's record, its family, and the leader's own entry in it.
    base = int(re.search(rb"# sound id\r\n\t\t\t(0x[0-9a-fA-F]+)", record).group(1), 16) & 0xFFFFFF
    if bfsar.user_field(arc.sound(base)["user"], "kind") != KIND:
        raise BuildError(f"the punch grunt (sound {base:#x}) does not choose by who makes it the way expected")
    slots = bfsar.TYPE_SLOTS[KIND]
    pattern = picks(arc, base, slots[LEADER])[0][0]
    like = arc.sound(pattern)
    template = arc.raw(bfsar.SOUNDS, pattern)
    if like["kind"] != bfsar.WAVE_SOUND:
        raise BuildError("the leader's punch grunt is not a wave sound")

    # New files: a wave archive holding his sounds, each once however many lines use it, and one
    # short silence; and the sounds that play them.
    files, archives, sounds = arc.count(bfsar.FILES), arc.count(bfsar.WAVE_ARCHIVES), arc.count(bfsar.SOUNDS)
    order: list[str] = []
    made: dict[str, tuple[list[int], int]] = {}
    for line in said:
        for take in line.takes:
            if take.name not in made:
                made[take.name] = (list(take.samples), take.rate)
                order.append(take.name)
            elif made[take.name] != (list(take.samples), take.rate):
                raise BuildError(f"two different sounds are both called {take.name}")
    for swap in own:
        take = swap.take
        if made.setdefault(take.name, (list(take.samples), take.rate)) != (list(take.samples), take.rate):
            raise BuildError(f"two different sounds are both called {take.name}")
        if take.name not in order:
            order.append(take.name)
    waves = [bfsar.make_wave(*made[name]) for name in order]
    waves.append(bfsar.make_wave([0] * SILENCE, SILENCE_RATE))
    silence = len(order)
    wave_archive = bfsar.make_wave_archive(waves)
    wave_sounds = bfsar.make_wave_sounds([(WAVE_ARCHIVE_ITEM | archives, i) for i in range(len(waves))])
    new_group = grp.append([(files, wave_archive), (files + 1, wave_sounds)])

    # New entries in the archive. A file entry and a wave archive entry are copies of the pattern's
    # own (both say only "this is in the main group"); a sound entry is the pattern's with new values.
    holder = arc.wave_archive(0)
    if (
        arc.file(like["file"]) != arc.file(holder)
        or arc.file(holder)["groups"] != MAIN_GROUP
        or len(arc.raw(bfsar.WAVE_ARCHIVES, 0)) != 12
    ):
        raise BuildError("the files of the punch grunt are not recorded the way expected")
    file_entry = arc.raw(bfsar.FILES, like["file"])
    archive_entry = struct.pack("<I", files) + arc.raw(bfsar.WAVE_ARCHIVES, 0)[4:]
    plain = bfsar.with_fields(
        like["user"],
        volume_spread=0,
        pitch_spread=0,
        variants=0,
        weight=1,
        spaced=0,
        gap=0,
        gap_spread=0,
        kind=0,
    )
    entries, bases = [], []
    for line in said:
        variants = len(line.takes) if len(line.takes) > 1 else 0
        bases.append(sounds + len(entries))
        for slot in range(max(slots.values()) + 1):
            for k, take in enumerate(line.takes):
                if slot == slots[LEADER]:
                    user = bfsar.with_fields(plain, weight=take.weight)
                    if k == 0:
                        user = bfsar.with_fields(
                            user,
                            variants=variants,
                            spaced=1 if line.gap or line.gap_spread else 0,
                            gap=line.gap,
                            gap_spread=line.gap_spread,
                        )
                    fields = {"index": order.index(take.name), "volume": VOLUME, "user": user}
                else:
                    fields = {"index": silence, "volume": 0, "priority": 0, "user": plain}
                if slot == 0 and k == 0:  # the family's first sound says how it is laid out
                    fields["user"] = bfsar.with_fields(fields["user"], kind=KIND, variants=variants)
                entries.append(bfsar.wave_sound_entry(template, file=files + 1, **fields))
    new_archive = arc.grow(
        {
            bfsar.SOUNDS: entries,
            bfsar.FILES: [file_entry, file_entry],
            bfsar.WAVE_ARCHIVES: [archive_entry],
        },
    )

    # The game's own sounds that are to play his: each one's entry, where it now is, sent to the new
    # file and the wave in it, at its volume. The small changes of loudness and pitch the game makes
    # at random to its own voice lines are taken off, as his other lines have none.
    grown, patched, done = bfsar.Archive(new_archive), bytearray(new_archive), set()
    named = bfsar.labels(archive)
    for swap in own:
        index = stock_sound(named, swap.label)
        was = arc.sound(index)
        if was["kind"] != bfsar.WAVE_SOUND or index in done:
            raise BuildError(f"{swap.label} is not a recording, or is asked for twice")
        done.add(index)
        at, size = grown.entry(bfsar.SOUNDS, index)
        try:
            patched[at : at + size] = bfsar.wave_sound_entry(
                arc.raw(bfsar.SOUNDS, index),
                file=files + 1,
                index=order.index(swap.take.name),
                volume=swap.volume,
                user=bfsar.with_fields(was["user"], volume_spread=0, pitch_spread=0),
            )
        except ValueError as error:
            raise BuildError(f"{swap.label}: {error}") from error
    new_archive = bytes(patched)

    # The new records of the table: the pattern's, each for another clip and frame and its own family.
    tag = re.search(rb'# anim tag\r\n\t\t\t"([^"]*)"', record).group(1)
    new_table = table[:-3].replace(
        b"# record num\r\n%d\r\n" % count,
        b"# record num\r\n%d\r\n" % (count + len(said)),
        1,
    )
    for i, line in enumerate(said):
        clip = line.clip.encode()
        if line.name.encode() in names:
            raise BuildError(f"the sound table already has a record named {line.name}")
        new = _swap_once(record, b'"' + PATTERN.encode() + b'"', b'"' + line.name.encode() + b'"')
        new = _swap_once(new, b"(STATE) " + tag + b" ", b"(STATE) " + clip + b" ")
        new = _swap_once(new, b'# anim tag\r\n\t\t\t"' + tag + b'"', b'# anim tag\r\n\t\t\t"' + clip + b'"')
        sound = b"0x%08x # the voice of leader %s" % (SOUND_ITEM | bases[i], LEADER.encode())
        new = _set_field(new, b"sound id", sound, old=rb"[^\r]*")
        new = _set_field(new, b"key id", b"%d" % (top_key + 1 + i))
        new = _set_field(new, b"frame", b"%d" % line.frame)
        new_table += new
    new_table += b"}\r\n"

    logger.info(
        "  %d sounds and a silence added as files %d and %d; sounds %#x to %#x play them",
        len(order),
        files,
        files + 1,
        sounds,
        sounds + len(entries) - 1,
    )
    logger.info("  %d of the leader's own sounds play his in place of theirs", len(own))
    return new_archive, new_group, new_table


def build(link: Link, romfs: RomFs, mod: Mod) -> None:
    """Write the sound archive, the group file and the command table into the mod folder."""
    archive, group, commands = romfs.read(ARCHIVE), romfs.read(GROUP), romfs.read(COMMANDS)
    table = sarc.Sarc(yaz0.decompress(commands)).files()[TABLE]
    voice = Voice(link.rom)
    recordings = cache(partial(shouts, voice))  # (each number of his voice read from the ROM once)
    new_archive, new_group, new_table = add_voice(
        archive,
        group,
        table,
        lines(voice, recordings, link),
        own_voice(recordings, link, archive, group),
    )
    mod.write(ARCHIVE, new_archive)
    mod.write(GROUP, new_group)
    mod.write(COMMANDS, repack(commands, {TABLE: new_table}))
