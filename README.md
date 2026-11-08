# Link in Pikmin 3

A mod for **Pikmin 3 Deluxe** that replaces Captain Alph with adult Link from **The Legend of
Zelda: Ocarina of Time**: his model, his animations and his voice, with the Master Sword, the
Hylian Shield, bombs, and bow.

> [!IMPORTANT]
> This repository does not contain the mod, or anything from either game. It contains a program
> that builds the mod from your own Ocarina of Time ROM and your own copy of Pikmin 3 Deluxe.

## Features

- **Link, straight from the ROM.** Ocarina of Time's model and textures, green tunic and all.
- **His own animations.** He walks, runs and stands the way he does in Ocarina, sword and shield out.
- **Sword combos and the spin attack.** Chains of swings that end in a bigger finishing blow, and a
  charged spin that hits everything around him. Even with sword trails.
- **His dodge roll.**
- **Bombs.** Ocarina's bomb, held over his head, thrown or set down. It goes off like a bomb rock.
- **The bow.** Aim on the move and fire arrows that hit the first enemy in their path.
- **His voice.** Battle cries as he throws and swings, and his own voice for most of what Alph
  used to say.
- **The HUD and menus** show Link wherever they showed Alph.

Brittany, Charlie, Louie, Olimar and the Pikmin are untouched.

## Controls

| Button | What Link does |
|---|---|
| **A** (no Pikmin following) | Swing the sword. Press again mid-swing to chain the next: stick released swings across, stick pushed swings down |
| **Hold A** | Charge, then let go for a spin attack |
| **D-pad Down** | Take out a bomb. **A** throws it, **D-pad Down** again sets it down |
| **Right stick click** | Take out the bow. **A** fires, another click puts it away |

With Pikmin following, A throws them as usual. Lock-on works with the bow out, too.

## Building the mod

You need **Python 3.10** or newer, and nothing else: there are no packages to install. You also
need:

1. **An Ocarina of Time ROM**, dumped from your own cartridge, in any byte order (`.z64`, `.n64`,
   `.v64`). The mod was made with the US 1.0 release; other versions are untested.
2. **Pikmin 3 Deluxe's RomFS**, extracted from your own copy of **version 1.2.0**. The new moves
   are a patch of that version's program, and a loader won't apply it to any other.

Then:

```bash
git clone https://github.com/rheard/pikmin-3-link
cd pikmin-3-link
python -m link "Ocarina of Time.z64" path/to/romfs
```

The build takes a few minutes and writes the mod, about 166 MB, to `build/Link` (`-o` writes it
somewhere else).

## Installing

The `Link` folder has two halves: `romfs`, the game files it replaces, and `exefs`, his moves'
code (a program file of its own, `subsdk1`, and a patch of the game's program that calls it).

- **On an emulator**, copy the whole `Link` folder into the game's mod folder. Emulators load both
  halves from there.
- **On a Switch running Atmosphère**, the pieces go to different places on the SD card:
  - everything in `Link/romfs/` goes in `atmosphere/contents/<title ID>/romfs/`, where
    `<title ID>` is Pikmin 3 Deluxe's sixteen-digit title ID;
  - `Link/exefs/subsdk1` goes in `atmosphere/contents/<title ID>/exefs/`;
  - the patch, `Link/exefs/D467F5AD367BBEE88C51F8C468143CF23486A44B.ips`, goes in a folder of its
    own under `atmosphere/exefs_patches/`, for example
    `atmosphere/exefs_patches/Link/D467F5AD367BBEE88C51F8C468143CF23486A44B.ips`. Keep its name:
    it is the build ID of the program the patch fits.

Then start the game. To get Alph back, take the files out again. Saves are not touched.

Link and [Mario](https://github.com/rheard/pikmin-3-mario) replace the same files, so use one at a
time.

## Known issues

- Cutscenes still show Alph, and a few of his lines are still Alph's.
- Anything Link has no animation of his own for (throwing Pikmin, plucking, whistling, being
  knocked down) plays Alph's, in which his arms and legs stay rigid.
- His eyes don't blink and his mouth doesn't move.

## TODO:

- Link's supercharged organge spin attack
- Blocking with the Hylian Shield.

## What's in here

| Path | What it does |
|---|---|
| `link/oot/` | Reads the ROM: Link's model, animations and voice, his bomb, his arrow, the spin's ring and the sword's trail |
| `link/formats/` | The Switch's file formats: archives, models, textures, sounds, effects |
| `link/character.py` | Link at this game's scale, with every clip of his laid out frame by frame |
| `link/model.py`, `link/bomb.py` | The new skeleton, meshes, textures and clips; the bomb's model |
| `link/moves.py`, `link/asm/` | The new moves, in ARM assembly, and `link/arm.py`, the assembler that builds them |
| `link/voice.py` | What he says, and when |
| `link/icons.py`, `link/render.py` | The HUD and menu pictures |

## The small print

This is a fan project. It is not made, endorsed or supported by Nintendo, and Link, Pikmin and
everything about them belong to Nintendo. The code here is original and is offered under the
[MIT licence](LICENSE); it contains nothing taken from either game. You need your own legally
obtained copies of both games to use it.
