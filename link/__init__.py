"""
Turn Alph into Link.

Builds a Pikmin 3 Deluxe mod from two things only you can supply: your own Ocarina of Time ROM and
the files of your own copy of Pikmin 3 Deluxe. Nothing of either game is kept in this repository.
"""

from __future__ import annotations

import logging

from typing import TYPE_CHECKING

from link import bomb, icons, model, moves, voice
from link.character import Character
from link.oot.actors import Bomb
from link.oot.link import Link
from link.oot.rom import Rom
from link.romfs import Mod, RomFs

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def build(rom: str | Path, romfs: str | Path, out: str | Path) -> None:
    """
    Build the whole mod.

    Args:
        rom: An Ocarina of Time ROM, in any of the three byte orders.
        romfs: Pikmin 3 Deluxe's extracted RomFS.
        out: The mod folder to write.
    """
    rom, romfs, mod = Rom(rom), RomFs(romfs), Mod(out)
    logger.info("reading Link from %s", rom.title.title())
    char = Character(rom)
    plain = Link(rom)  # with his hands open, as he holds a bomb
    logger.info("the model, his clips and the effects")
    clips = model.build(char, romfs, mod)
    logger.info("his bomb")
    radius = bomb.build(Bomb(plain), char.scale, romfs, mod)
    logger.info("his code")
    moves.build(char, clips, plain, radius, mod)
    logger.info("his voice")
    voice.build(plain, romfs, mod)
    logger.info("his pictures")
    icons.build(rom, romfs, mod)
    logger.info("done: the mod is in %s", mod.root)
