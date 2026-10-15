"""
Link as the model builder wants him: his triangles in the game's units, the bone each part follows,
and every clip of his own, laid out frame by frame from his animations.

He has the Master Sword in his left hand and the Hylian Shield on his right arm, and stands, walks
and runs as his own game has him with both out. What he holds changes as he moves, as it does in his
own game: his model has three sets of what his scabbard and hands can be drawn as (`sets`), each a
"look" on a small bone of its own that a clip shows or shrinks to nothing. With a bomb in his hands
the sword and shield go on his back; with the bow out his right hand holds the bow.

His clips, in the order the code counts on them (`Character.moves`):

* **Swings** (`SWINGS`): his game's swing across from his right, twice, and its later blow; then the
  swing straight down, twice, and its later blow. In his game the stick decides which: let go, the
  first kind; pushed forward, the second. Each is the swing and then his way back to his stand.
* **His roll** (`DODGES`), where the game's dodge rolls a leader.
* **A bomb** (`BOMBS`): he raises his hands over his head, holds a bomb there, throws it, or sets it
  down in front of him.
* **Left standing** (`IDLES`): five things he does in his game when left alone, and his sitting down
  on the ground, sitting, and getting up again.
* **His bow** (`BOWS`): raising it, aiming, loosing an arrow (which flies on, a look of its own) and
  taking the next arrow.
* **His spin** (`SPINS`): his game's charge of the sword, the spin it ends in (with his game's ring
  of light round him) and his straightening up when it is let go too soon.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from itertools import pairwise
from operator import itemgetter
from typing import TYPE_CHECKING

from link.errors import BuildError
from link.geometry import Matrix, Vector, apart, euler_matrix, matmul, rotate, transpose
from link.oot.actors import Arrow, SpinRing, SwordTrail
from link.oot.animation import PARTS, Animation, Keys, Mixed, Sequence
from link.oot.link import ARMED, Link, Pose

if TYPE_CHECKING:
    from link.oot.animation import Mix, Read, Value
    from link.oot.graphics import Image, Material
    from link.oot.rom import Rom

# A triangle's corner: position, normal, (u, v) with v up, material, and the part it follows.
Corner = tuple[Vector, Vector, tuple[float, float], "Material", str]

HEIGHT = 26.0  # how tall he stands here, to the top of his hat: Alph is 16.3 with his helmet on
RATE = 1.5  # frames of this game to one of his: it runs at 30 a second, his at 20

# The leader bone each of his parts follows, and the joint of his each of those bones must sit on.
STOCK = {
    "waist": "sebonjnt", "torso": "sebonjnt", "collar": "sebonjnt", "sheath": "sebonjnt",
    "head": "headjnt", "hat": "headjnt",
    "left_upperarm": "lhandjnt", "left_forearm": "lhandjnt", "left_hand": "lhandjnt",
    "right_upperarm": "rhandjnt", "right_forearm": "rhandjnt", "right_hand": "rhandjnt",
    "left_thigh": "llegjnt", "left_shin": "llegjnt", "left_foot": "llegjnt",
    "right_thigh": "rlegjnt", "right_shin": "rlegjnt", "right_foot": "rlegjnt",
}  # fmt: skip
PIVOTS = {
    "sebonjnt": "waist", "headjnt": "head", "lhandjnt": "left_upperarm", "rhandjnt": "right_upperarm",
    "llegjnt": "left_thigh", "rlegjnt": "right_thigh",
}  # fmt: skip
UPPER_STOCK = "sebonjnt"  # the leader's spine
UPPER_BONES = ("sebonjnt", "headjnt", "lhandjnt", "rhandjnt")  # the leader's bones above his legs
# A limb that never turns against another in any of his animations needs no bone of its own: its
# corners follow the other's. His collar and his trunk both hang from the joint of his upper body,
# and neither is given an angle of its own.
MERGE = {"collar": "torso"}

# What his scabbard and hands are drawn as (the program's own kinds of list): the sword and shield
# out; put away on his back with his hands open, while he holds a bomb; and the bow in his right
# hand, the sword and shield on his back.
LOOK_PARTS = ("sheath", "left_hand", "right_hand")
DRAWN = ("sheath empty", "left sword", "right shield")
AWAY = ("sheath sword shield", "left open", "right open")
BOW = ("sheath sword shield", "left closed", "right bow")

# The leaders' clips his own walk and run ride in, by how fast a leader is going: `asibumi` below a
# speed of 15, `walk` to 45, `run2` above, and `nigeru` above 85, which no leader reaches in play.
# His walk covers the ground at about the speed `asibumi` plays at, so it is used for both slow ones.
CLIPS = {"asibumi": "walk" + ARMED, "walk": "walk" + ARMED, "run2": "run" + ARMED, "nigeru": "run" + ARMED}


@dataclass(frozen=True)
class Stand:
    """A clip of his own that plays where every leader plays one of the game's (`player_wait`)."""

    tag: str  # the entry of the game's list of clips that is pointed at it
    like: str  # the stock clip that entry names, which the other leaders go on seeing
    pieces: tuple[str, ...]  # his animations, played one after another and round again
    rate: float  # frames of this game to one of his


# Standing still: his stand twice, then his look about him and behind him, as his game has him do
# when left alone, and round again.
STANDS = {"l_wait": Stand("player_wait", "wait", ("wait" + ARMED, "wait" + ARMED, "look" + ARMED), RATE)}

# His swings: (clip, the kind of swing in his game's table). The first two of a kind are one swing
# under two names: the game is told to play a clip by its number, and the second press has to name
# another. A kind's three together, the stick-let-go kind first.
SWINGS = (
    ("l_slash1", "right"), ("l_slash2", "right"), ("l_slash3", "right_combo"),
    ("l_chop1", "forward"), ("l_chop2", "forward"), ("l_chop3", "forward_combo"),
)  # fmt: skip
RUN = 3  # swings in a run of one kind
BLADE = 2.0 / 3.0  # how far along the sword, from his hand to its tip, the spark of a hit is drawn
# Where a swing made with the stick forward carries him. By the public documentation of his game's
# source his speed is set to 15 of its units a frame as the swing starts and falls by 5 a frame, so
# he goes 10 and then 5: that is 1,500 of his model's units, over two of his frames.
LUNGE, LUNGE_FRAMES = 1500.0, 2
WALK = 160.0  # a leader's speed with the stick right over, units a second: what a step's speed is a share of

# His roll, played once where the game's dodge plays the leaders' roll forward.
DODGES = (("l_roll", "roll"),)

# A bomb held over his head: his lift (from the first frame he is upright), his hold, his throw,
# and his setting it down, by his game's own animation of it at twice his speed.
BOMBS = (("l_bomb_up", "lift"), ("l_bomb_hold", "hold"), ("l_bomb_throw", "throw"), ("l_bomb_put", "put"))
UPRIGHT = 0.95  # how near upright his trunk must be for his lift to start from that frame (1 is upright)
ARRIVED = 0.9  # how much of the way over his head his hands must be for the bomb to be made between them
BOMB_EASE = 4  # frames of his over which he settles from the lift into the hold
HOLD_LIKE = "trwwait"  # the leaders' clip for holding a Pikmin, whose header a clip that goes round takes
PUT_EASE = 2  # frames of his over which he is eased from the hold into setting it down
PUT_SETTLE = 2  # ... and over which the end of setting it down is eased into his stand
PUT_UPRIGHT = 0.9  # how near upright his trunk is until he bows over what he has set down
PUT_SPEED = 2.0  # how many times his own speed his setting it down is played at
ARMS = ("left_upperarm", "left_forearm", "left_hand", "right_upperarm", "right_forearm", "right_hand")
BACK = 60.0  # degrees: how near his stand his arms must be for the sword and shield to be in his hands again
# Where a ball sits in his hands (`bomb_at`): out beyond them along the line his arms lie on, its
# surface `HAND_CLEAR` units past where his hands end; and, as he sets it down, no lower than
# `PUT_DROP` above the level of his feet, from where it is let go.
HAND_CLEAR = 1.0
PUT_DROP = 3.0

# Left standing: five things he does in his game when left alone with the sword and shield out, each
# eased in from his stand and back into it, played in turn; then his sitting down on the ground,
# laid out as the leaders' sitting clips are (it goes round its middle until told to finish).
IDLES = (
    ("l_glance", "glance"), ("l_stretch", "stretch"), ("l_boot", "boot"), ("l_shield", "shield"),
    ("l_flourish", "flourish"), ("l_sit", "sit"),
)  # fmt: skip
TURNS = 5  # how many of those the code plays in turn
IDLE_EASE = 6  # frames of his over which a thing he does left standing is eased in from his stand, and back into it
SIT_EASE = 4  # ... and over which sitting down is eased into sitting, and sitting into getting up

# His bow: raising it, aiming, loosing and taking the next arrow.
BOWS = (("l_bow_ready", "ready"), ("l_bow_aim", "aim"), ("l_bow_loose", "loose"), ("l_bow_next", "next"))
BOW_EASE = (
    3  # frames of his over which raising the bow is eased in from his stand, and taking the next arrow from loosing
)
BOW_SETTLE = 2  # ... and over which the end of taking the next arrow is eased into the aim
ARROW_SPEED = 20.0  # units of this game a frame (600 a second)
ARROW_FRAMES = 30  # frames it flies for
ARROW_LANDS = 6.0  # how far above the level of his feet its point is at the end of that
ARROW_SIZE = 1.25  # how many times its size in his game the arrow is, about its nock, so that it is seen in flight
ARROW_SHOWS = 3  # the frame of `l_bow_ready` from which the arrow is in his hand
SHAFT_ROWS = 3  # the arrow's shaft is a long strip for a thing a third of a unit wide: every third row is kept

# A charge of his sword: the way into the stance and the stance going round; his spin and his way
# back to his stand; and his straightening up, let go too soon.
SPINS = (("l_charge", "charge"), ("l_spin", "spin"), ("l_lower", "lower"))
CHARGE_EASE = 4  # frames of his over which the end of the way into the stance is eased into the stance

# The ring of light his spin throws out: his game's own two lists (a flat ring, and a band standing
# up round it) on a part of its own, `ring`, as three looks: the flat ring's inner edge, its outer
# edge, and the band. Each is given its size on every frame of `l_spin`, as his game grows it and
# thins it away. It is drawn by the leaders' antenna bulb's material (Link has no antenna), told to
# draw see-through as their helmet glass does, with his game's picture of it drawn into a texture
# of Alph's eyes that nothing on Link draws.
RING_LOOKS = ("ring_in", "ring_out", "ring_band")
RING_MATERIAL = "light_m"
RING_PICTURE = "playerA_eye_spq.1"
RING_PICTURE_SIZE = (128, 64)  # each list's picture: round the ring, across it (the texture is 128 by 128)
RING_RENDER = {
    "gsys_render_state_mode": "translucent",
    "gsys_render_state_blend_mode": "color",
    "gsys_depth_test_write": "false",
    "gsys_render_state_display_face": "front",
    "pass": "CrossWater",
}
RING_PARAMS = {"spec_rate": (0.0,), "rim_color": (0.0, 0.0, 0.0)}  # no shine, no light on its rim
HIDDEN = 0.001  # the size a clip gives the bone of a look that is not to show

# The trail his sword leaves as he swings it, as his game makes and draws it (`SwordTrail`): a point
# of the blade at each edge added on each frame of a swing, and drawn for three of his frames after,
# white, fading. On his model it is a ribbon of as many points along it as his game's smoothed
# drawing has, each on a bone of its own (`TRAIL_SLOT`) under a part (`TRAIL_PART`) and its one look
# (`TRAIL_LOOK`), which shows in the clips of his swings and his spin alone, and only those clips
# move the points' bones. On each frame of them, a point some frames old is where his hand was that
# long before; its picture is his game's colour at each age along it and across the blade. It is
# drawn by Alph's eyes' material (Link's eyes are drawn otherwise), given the antenna bulb's shader
# options and numbers so that it draws as the ring does; the animations of Alph's eyes, which swap
# that material's pictures, are pointed at the trail's picture alone.
TRAIL_PART, TRAIL_LOOK, TRAIL_SLOT = "trail", "streak", "trail_{:02d}"
TRAIL_MATERIAL = "eye_m"
TRAIL_LIKE = RING_MATERIAL
TRAIL_PICTURE = "playerA_eye_spq.2"
TRAIL_PICTURE_SIZE = (128, 128)  # along the trail, from the blade to the oldest; across it, from the tip edge
TRAIL_ANIMS = ("damageA_ftp", "mabatakiA_ftp")  # Alph's eyes when hurt, and blinking


@dataclass
class Move:
    """
    A clip of his own, which no clip of the leaders' stands behind.

    Attributes:
        family: What it is part of: "swing", "dodge", "bomb", "idle", "bow" or "spin".
        kind: What he does in it, within that.
        count: Its last frame.
        times: For each frame, the frame it shows of what it is laid out from.
        hit: The frame on which the punch state looks for something to hit, if it marks one.
        more: The frame the code acts on (the next press, the bomb made or let go), if it marks one.
        loop: The two frames it goes round between, if it goes round.
        window: For the spin, the frames on which the blade can hit.
        spark: The part the spark of a hit belongs on, and the frame (or frames) to find it at.
        step: How far a swing carries him: for how many frames, at how many units a second.
        like: The stock clip whose header it takes; one that goes round takes a looping one's.
        shows: For each frame, the looks that show, where that is not the sword and shield out.
        sizes: For each frame, the size of looks that grow and thin (the ring's).
        arrow: For each frame of a clip of the bow, where the arrow is: "nocked", "flying" or None.
    """

    family: str
    kind: str
    count: int
    times: list[float]
    hit: int | None = None
    more: int | None = None
    loop: tuple[int, int] | None = None
    window: tuple[int, int] | None = None
    spark: tuple[str, int | tuple[int, int]] | None = None
    step: tuple[int, float] = (0, 0.0)
    like: str = "punch"
    shows: list[tuple[str, ...]] | None = None
    sizes: dict[str, list[Vector]] | None = None
    arrow: list[str | None] | None = None


# -- what his clips are laid out from ----------------------------------------------------------------


class SetDown(Mixed):
    """
    His setting down what he holds over his head, by his own animation of it. He is eased into it
    from his hold over `PUT_EASE` frames, and its last `PUT_SETTLE` frames are eased into the first
    frame of his stand. Past its end, its last frame.
    """

    def __init__(self, link: Link) -> None:
        self.hold, self.put, self.stand = (link.animation(which) for which in ("hold", "put" + ARMED, "wait" + ARMED))
        self.put.loops = False
        self.frames, self.last = self.put.frames, self.put.frames - 1.0
        if self.last <= PUT_EASE + PUT_SETTLE:
            raise BuildError(f"his setting a thing down is too short to ease into and out of ({self.frames} frames)")
        self.ease, self.settle, self.held = float(PUT_EASE), float(PUT_SETTLE), float(BOMB_EASE)

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        t = min(max(float(frame), 0.0), self.last)
        now = what(self.put, t)
        if t < self.ease:
            return mix(what(self.hold, self.held), now, t / self.ease)
        if self.settle and t > self.last - self.settle:
            return mix(now, what(self.stand, 0.0), (t - (self.last - self.settle)) / self.settle)
        return now


class BowKeys(Mixed):
    """
    One of the four things he does with his bow. `ready` is his raising of it, eased in from the
    first frame of his stand with the sword out; `aim` his aiming, which comes round; `loose` the
    string hand flung back, its first frame eased in from the aim's and its last held; `next` his
    reaching forward and drawing again, eased in from where `loose` ends and into the aim's first
    frame. `twist(frame)` is how much of the turn of his upper body the frame has: all of it, but as
    he raises the bow, where it comes in with the easing.
    """

    def __init__(self, link: Link, kind: str) -> None:
        self.kind = kind
        aim, self.main = link.animation("bow_aim"), link.animation("bow_" + kind)
        self.main.loops = kind == "aim"
        self.frames, self.last = self.main.frames, self.main.frames - 1.0
        self.before, self.after, self.ease, self.settle = None, None, 0.0, 0.0
        if kind == "ready":
            self.before, self.ease = (link.animation("wait" + ARMED), 0.0), float(BOW_EASE)
        elif kind == "loose":
            self.before, self.ease = (aim, 0.0), 1.0
        elif kind == "next":
            loose = link.animation("bow_loose")
            self.before, self.ease = (loose, loose.frames - 1.0), float(BOW_EASE)
            self.after, self.settle = (aim, 0.0), float(BOW_SETTLE)
        if self.ease + self.settle >= self.last and kind != "aim":
            raise BuildError(f"his {kind} with the bow is too short to ease into and out of ({self.frames} frames)")

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        if self.kind == "aim":
            return what(self.main, float(frame) % self.last)  # (its last frame is its first)
        t = min(max(float(frame), 0.0), self.last)
        now = what(self.main, t)
        if self.before and t < self.ease:
            return mix(what(*self.before), now, t / self.ease)
        if self.after and t > self.last - self.settle:
            return mix(now, what(*self.after), (t - (self.last - self.settle)) / self.settle)
        return now

    def twist(self, frame: float) -> float:
        """How much of his turn towards where he aims the frame has."""
        return min(max(float(frame), 0.0) / self.ease, 1.0) if self.kind == "ready" else 1.0


class Fidget(Mixed):
    """
    One of the things he does when left standing: the animation whole, its first `IDLE_EASE` frames
    eased in from the first frame of his stand and its last eased into it. Past its end, its last
    frame, which is the stand's first.
    """

    def __init__(self, link: Link, which: str) -> None:
        self.stand, self.does = (link.animation(name) for name in ("wait" + ARMED, "idle_" + which))
        self.does.loops = False
        self.frames, self.last, self.ease = self.does.frames, self.does.frames - 1.0, float(IDLE_EASE)
        if self.last <= 2 * IDLE_EASE:
            raise BuildError(f"his {which} is too short to ease into and out of ({self.frames} frames)")

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        t = min(max(float(frame), 0.0), self.last)
        now = what(self.does, t)
        if t < self.ease:
            return mix(what(self.stand, 0.0), now, t / self.ease)
        if t > self.last - self.ease:
            return mix(now, what(self.stand, 0.0), (t - (self.last - self.ease)) / self.ease)
        return now


class Sit(Mixed):
    """
    His sitting down on the ground, sitting and getting up: three animations of his laid end to end.

    Going down (`down` frames, from the last frame of it on which his hips are at their highest) is
    eased in from the first frame of his stand over `IDLE_EASE` frames and, over its last `SIT_EASE`,
    into the first frame of sitting; sitting is that animation once round (`round` frames: at its
    end he is exactly where it began); getting up (`up` frames, to the first frame his hips are back
    at nearly their highest, and `IDLE_EASE` more) is eased in from sitting over `SIT_EASE` and into
    his stand over its last `IDLE_EASE`.

    The three are made for a shorter body, young Link's (`Link.sits`): the root's height in them is
    divided by the share that gives, which puts his feet on the ground as he stands and his seat on
    it as he sits. And his collar, which shares his trunk's bone (`MERGE`), is held to his trunk as
    his rest pose has it: in these three it is half a degree off, where his own never turn it.

    Raises:
        BuildError: If the three are not found, or do not join up.
    """

    def __init__(self, link: Link) -> None:
        found, self.share = link.sits()
        if not found:
            raise BuildError("his sitting down, sitting and getting up were not found among his animations")
        self.stand, self.going, self.stay, self.rise = (
            link.animation(name) for name in ("wait" + ARMED, "sit_down", "sit", "sit_up")
        )

        def height(a: Animation) -> list[float]:
            return [a.root(float(n))[1] for n in range(a.frames)]

        going, rising = height(self.going), height(self.rise)
        first = max(n for n, y in enumerate(going) if y >= 0.995 * max(going))
        low = rising.index(min(rising))
        back = min(n for n, y in enumerate(rising) if n > low and y >= 0.98 * max(rising))
        self.going.loops = self.rise.loops = False
        self.stay.loops = True
        self.first = float(first)
        self.down, self.round = self.going.frames - 1.0 - self.first, self.stay.frames - 1.0
        self.up = float(back + IDLE_EASE)
        self.ease, self.settle = float(SIT_EASE), float(IDLE_EASE)
        if (
            min(self.down, self.up) <= SIT_EASE + IDLE_EASE
            or self.up > self.rise.frames - 1.0
            or apart(self.stay.turns(self.round), self.stay.turns(0.0)) > 2.0
        ):
            raise BuildError("his sitting down does not come out as three animations to join")
        rest = dict(zip(PARTS, link.pose(), strict=False))
        self.held = {part: (host, matmul(transpose(rest[host][0]), rest[part][0])) for part, host in MERGE.items()}

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        t = min(max(float(frame), 0.0), self.down + self.round + self.up)
        there = what(self.stay, 0.0)
        if t <= self.down:
            now = what(self.going, self.first + t)
            if t < self.settle:
                return mix(what(self.stand, 0.0), now, t / self.settle)
            return now if t <= self.down - self.ease else mix(now, there, (t - (self.down - self.ease)) / self.ease)
        if t < self.down + self.round:
            return what(self.stay, t - self.down)
        v = t - self.down - self.round
        now = what(self.rise, v)
        if v < self.ease:
            return mix(there, now, v / self.ease)
        return (
            now
            if v <= self.up - self.settle
            else mix(now, what(self.stand, 0.0), (v - (self.up - self.settle)) / self.settle)
        )

    def turns(self, frame: float) -> dict[str, Matrix]:
        """Every limb's rotation at a frame, his collar held to his trunk."""
        out = dict(super().turns(frame))
        for part, (host, against) in self.held.items():
            out[part] = matmul(out[host], against)
        return out

    def root(self, frame: float) -> Vector:
        """Where the root is at a frame, its height put right for his body."""

        def tall(a: Animation, u: float) -> Vector:
            return a.root(u) if a is self.stand else (a.root(u)[0], a.root(u)[1] / self.share, a.root(u)[2])

        return self._mix(frame, tall, lambda p, q, f: tuple(x + (y - x) * f for x, y in zip(p, q, strict=True)))


class Charge(Mixed):
    """
    His going into the charge's stance and holding it, on one line of his frames: the way in, from
    the frame his code starts it on to its last (`lead` frames); then the stance, eased in from that
    last frame over `CHARGE_EASE`; then the stance going round, from the frame the easing ended on,
    `round` frames, back to that frame. `going` is the frame the going round starts on, and `frames`
    the last frame of all, where it is back there.
    """

    def __init__(self, link: Link) -> None:
        found = link.spins()
        self.into, self.stance = (Animation(link, found[name]) for name in ("charge_in", "charge"))
        self.into.loops = False
        self.start = float(found["from"])
        self.lead = self.into.frames - 1.0 - self.start
        self.ease, self.round = float(CHARGE_EASE), float(self.stance.frames)
        self.going = self.lead + self.ease
        self.frames = self.going + self.round
        if not 0.0 < self.lead < self.into.frames or not 0.0 < CHARGE_EASE < self.stance.frames:
            raise BuildError("the way into his charge does not come out as a stretch to ease from")

    def _mix(self, frame: float, what: Read[Value], mix: Mix[Value]) -> Value:
        t = min(max(float(frame), 0.0), self.frames)
        if t <= self.lead:
            return what(self.into, self.start + t)
        u = t - self.lead
        if u < self.ease:
            return mix(what(self.into, self.into.frames - 1.0), what(self.stance, u), u / self.ease)
        return what(self.stance, u % self.round)


# -- Link --------------------------------------------------------------------------------------------


class Character:
    """
    Link, read from the ROM and stood in the game's units, with every clip of his own.

    Attributes:
        link: The model as his own game has it.
        tris: Every triangle at rest, as corners.
        frames: Where each part (and each thing a part can be drawn as) sits at rest.
        stock: The leader bone each part follows.
        pivot: Where each leader bone of the rig must sit.
        top: The highest point of his head.
        looks: {look: the part it is on}: the things his scabbard, hands, arrow and ring are drawn as.
        sets: The looks that show together: the sword and shield out, put away, and the bow.
        in_head: Looks whose triangles go in the head's shape, which has room to spare.
        glow: The ring, as a see-through shape of its own.
        streak: The trail of his sword, as another.
        trail: {"look", "slots", "clips"}: the look that shows the trail, the parts of its points, which
            have bones but no part of the leader's to hang from, and the clips that move them.
        moves: Every clip of his own, in the order the code counts on.
    """

    zooms = (4, 3.5, 3, 2.5, 2, 1.75, 1.5, 1.45, 1.4, 1.35, 1.3, 1.25, 1)  # sizes to try packing his textures at
    clips = CLIPS
    stands = STANDS

    def __init__(self, rom: Rom, height: float = HEIGHT) -> None:
        names = ("drawn", "away")
        self.sets = {name: tuple(f"{limb}_{name}" for limb in LOOK_PARTS) for name in names}
        self.looks = {look: limb for name in names for look, limb in zip(self.sets[name], LOOK_PARTS, strict=True)}
        self.shown = self.sets["drawn"]
        kinds = {
            look: kind
            for name, set_kinds in zip(names, (DRAWN, AWAY), strict=True)
            for look, kind in zip(self.sets[name], set_kinds, strict=True)
        }
        # (with the bow out the sword and shield are on his back, as with a bomb)
        self.sets["bow"] = (self.sets["away"][0], "left_hand_bow", "right_hand_bow")
        kinds.update(zip(self.sets["bow"][1:], BOW[1:], strict=True))
        self.looks.update(zip(self.sets["bow"][1:], LOOK_PARTS[1:], strict=True))
        # (the leader's body shape has 1,066 vertices and every set needs more: his open hands and his hands with the
        # bow go in the head's shape, which has room, and whose material is the one his face is drawn with)
        self.in_head = self.sets["away"][1:] + self.sets["bow"][1:]
        self.link = link = Link(rom, holding=DRAWN, looks=kinds)
        # The arrow: a part no animation of his moves, and on it one look, which shows in the bow's clips alone.
        self.arrow = Arrow(link)
        self.arrow.resize(ARROW_SIZE)
        self.nock = link.add_prop("nock", "arrow", self.arrow.tris)
        self.looks["arrow"] = "nock"
        self.small = {material: SHAFT_ROWS for material, _, _ in self.arrow.tris if link.image(material)[1] > 64}
        # where each fist's middle is in its hand's own space: the arrow lies from the one to the other
        self.fists = tuple(
            _middle(link.read_list(link.held(kind), PARTS.index(limb)))
            for kind, limb in (("left closed", "left_hand"), ("right closed", "right_hand"))
        )
        # The ring: a part no animation of his moves, and its three looks, which only the spin shows, all at once.
        self.ring = SpinRing(link)
        self.ring_part = link.add_part("ring")
        self.looks.update(dict.fromkeys(RING_LOOKS, "ring"))
        # The trail of his sword: a part no animation of his moves, its one look, which only the clips of his swings
        # and his spin show, and under that a bone for each point along the trail (`swing_moves` and `spin_moves` say
        # which clips).
        self.sword_trail = SwordTrail(link)
        count = round(self.sword_trail.span() * self.sword_trail.steps) + 1
        self.trail_part = link.add_part(TRAIL_PART)
        self.slot_parts = [link.add_part(TRAIL_SLOT.format(k)) for k in range(count)]
        self.looks[TRAIL_LOOK] = TRAIL_PART
        self.trail = {"look": TRAIL_LOOK, "slots": [TRAIL_SLOT.format(k) for k in range(count)], "clips": []}
        self._poses = {}
        rest = link.pose()
        skin = [
            (material, [(*c[:3], MERGE.get(c[3], c[3])) for c in tri])
            for material, _, tri in link.skin(rest, looks=True)
        ]
        # (his height is his own as his game draws him: the looks are measured out)
        ys = [c[0][1] for _, _, tri in link.skin(rest) for c in tri]
        self.source_height = max(ys) - min(ys)
        self.scale = height / self.source_height
        self.lift = -min(ys) * self.scale  # nothing: the rest pose stands on the ground, as his animations do
        self.tris = [
            [(self.place(p), n, (uv[0], 1.0 - uv[1]), material, part) for p, n, uv, part in tri]
            for material, tri in skin
        ]
        # (the arrow's part, the ring's and the trail's hang from the leader's spine, which rests in every clip they
        # show in)
        stock = dict(STOCK, nock=UPPER_STOCK, ring=UPPER_STOCK, trail=UPPER_STOCK)
        stock.update({look: stock[limb] for look, limb in self.looks.items()})
        used = {c[4] for tri in self.tris for c in tri}
        # (a limb that has looks has a bone as every limb has, and the looks' bones hang under it)
        limbs = (used - set(self.looks)) | set(self.looks.values())
        self.frames = {
            part.name: (rot, self.place(pos))
            for part, (rot, pos) in zip(link.parts, rest, strict=True)
            if part.name in limbs
        }
        self.stock = {name: stock[name] for name in used | limbs}
        self.pivot = {bone: self.place(rest[PARTS.index(part)][1]) for bone, part in PIVOTS.items()}
        self.top = max((c[0] for tri in self.tris for c in tri if self.stock[c[4]] == "headjnt"), key=itemgetter(1))
        self.glow = self._ring_glow()
        # (each point of the trail rests where his left hand does: at rest the ribbon is the blade's edge, and has no
        # breadth)
        hand = rest[PARTS.index("left_hand")]
        for name in self.trail["slots"]:
            self.frames[name] = (hand[0], self.place(hand[1]))
            self.stock[name] = UPPER_STOCK
        self.streak = self._trail_streak()
        self.moves = {}
        for moves in (
            self.swing_moves(),
            self.dodge_moves(),
            self.bomb_moves(),
            self.idle_moves(),
            self.bow_moves(),
            self.spin_moves(),
        ):
            self.moves.update(moves)

    def place(self, p: Vector) -> Vector:
        """A point of his model in the game's units."""
        return (p[0] * self.scale, p[1] * self.scale + self.lift, p[2] * self.scale)

    def image(self, material: Material) -> Image:
        """A material's picture; for a long thin strip, every so many rows of it."""
        w, _, rows = self.link.image(material)
        step = self.small.get(material)
        if step:
            rows = rows[::step]
        return w, len(rows), rows

    # -- the ring of his spin ------------------------------------------------------------------------

    def _ring_glow(self) -> dict:
        """
        The ring, as a see-through shape of its own.

        Its two lists at the size it grows to with his sword, about its part as that rests, flat
        (his game turns it about the upright with his facing, which a ring of equal sides round its
        middle does not show), each triangle twice, the second turned over, since the material draws
        the front of a triangle only (his game draws both). Every corner faces up: his game lights
        none of it. The inner corners of the flat ring follow `ring_in`, the outer `ring_out`, the
        band's `ring_band`. Each corner reads the texture where his game's reads its first tile:
        round the ring once across the picture's width; across the ring over its height, from the
        middle of its first row to the middle of its last, so that filtering never reaches the other
        list's picture. The two lists' pictures are drawn one over the other, in the lists' order.

        Returns:
            {"material", "tris", "params", "render", "texture"}, as the model's builder takes them.
        """
        ring = self.ring
        k = ring.size * ring.grown / ring.his_size * self.scale  # a list's units to this game's, at its full size
        centre = self.frames["ring"][1]
        flat = [math.hypot(p[0], p[2]) for tri in ring.flat for p in tri]
        middle = (min(flat) + max(flat)) / 2.0
        cols, lines = RING_PICTURE_SIZE
        rows, tris = [], []
        for n, address in enumerate(ring.lists):
            rows += ring.picture(address, cols, lines)
            source = ring.corners[address]
            ss, ts = [c[1][0] for tri in source for c in tri], [c[1][1] for tri in source for c in tri]
            s0, s1, t0, t1 = min(ss), max(ss), min(ts), max(ts)
            band = len({c[0][1] for tri in source for c in tri}) == 2
            for tri in source:
                corners = []
                for p, (s, t) in tri:
                    look = "ring_band" if band else "ring_in" if math.hypot(p[0], p[2]) < middle else "ring_out"
                    u = (s - s0) / float(s1 - s0)
                    v = (n * lines + 0.5 + (t - t0) / float(t1 - t0) * (lines - 1)) / float(lines * len(ring.lists))
                    corners.append(
                        (tuple(centre[i] + p[i] * k for i in range(3)), (0.0, 1.0, 0.0), (u, v), RING_MATERIAL, look),
                    )
                tris += [corners, corners[::-1]]
        return {
            "material": RING_MATERIAL,
            "tris": tris,
            "params": dict(RING_PARAMS),
            "render": dict(RING_RENDER),
            "texture": (0, RING_PICTURE, rows),
        }

    def ring_turn(self, n: int) -> Matrix:
        """
        The ring's turn about the upright on frame `n` of `l_spin`: the half turn from his facing his
        game makes it with, and back round by as far as his game has moved its first texture round
        it by then (the corners' coordinate round it grows the other way).

        Returns:
            The turn, as a rotation matrix.
        """
        a = self.ring.made_turned - self.ring.turned(n / RATE)
        return [[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]]

    def ring_sizes(self, count: int) -> dict[str, list[Vector]]:
        """
        The size of each of the ring's looks on each frame of `l_spin`, `count` + 1 of them.

        On frame k his game's ring is as it is k / `RATE` of his frames after the spin begins: each
        look is that size against the one it was built at, and where his game draws it in part, the
        flat ring is narrower about its middle and the band lower, in proportion; where it draws
        none, all three are `HIDDEN`.

        Returns:
            Each look's (x, y, z) size on each frame.
        """
        ring = self.ring
        flat = [math.hypot(p[0], p[2]) for tri in ring.flat for p in tri]
        inner, outer = min(flat), max(flat)
        middle, half = (inner + outer) / 2.0, (outer - inner) / 2.0
        out = {look: [] for look in RING_LOOKS}
        for n in range(count + 1):
            size, drawn = ring.frame(n / RATE)
            if drawn <= 0.0:
                for look in RING_LOOKS:
                    out[look].append((HIDDEN,) * 3)
                continue
            s = size / ring.grown
            out["ring_in"].append((s * (middle - half * drawn) / inner,) * 3)
            out["ring_out"].append((s * (middle + half * drawn) / outer,) * 3)
            out["ring_band"].append((s, s * drawn, s))
        return out

    # -- the trail of his sword ----------------------------------------------------------------------

    def _trail_streak(self) -> dict:
        """
        The trail, as a see-through shape of its own: a ribbon along the trail's points, each with a
        corner at the blade's tip edge and one at its other edge where his game's trail has them, each
        square of it twice, the second turned over (his game draws both sides, the material one). The
        picture is his game's trail's colour at each age along it and each place across it; a point's
        corners read it at the point's age, from the middle of the first column (the point just added)
        to the middle of the last (the oldest drawn), the tip edge in the middle of the first row and
        the other in the last's. Every corner faces up, as the ring's do.

        Returns:
            {"material", "like", "tris", "params", "render", "texture", "anims"}, as the model's builder
            takes them.
        """
        trail, slots = self.sword_trail, self.trail["slots"]
        rot, pos = self.frames[slots[0]]

        def at(p: Vector) -> Vector:
            return tuple(pos[k] + c * self.scale for k, c in enumerate(rotate(rot, p)))

        edges = (at(trail.blade_tip), at(trail.blade_base))
        cols, lines = TRAIL_PICTURE_SIZE
        last = len(slots) - 1

        def corner(k: int, edge: int) -> Corner:
            uv = ((0.5 + (cols - 1) * k / float(last)) / cols, (0.5 + (lines - 1) * edge) / lines)
            return edges[edge], (0.0, 1.0, 0.0), uv, TRAIL_MATERIAL, slots[k]

        tris = []
        for k in range(last):
            a, b, c, d = corner(k, 0), corner(k + 1, 0), corner(k + 1, 1), corner(k, 1)
            for tri in ([a, b, c], [a, c, d]):
                tris += [tri, tri[::-1]]
        rows = [
            [
                tuple(round(v) for v in trail.colour(trail.span() * x / float(cols - 1), y / float(lines - 1)))
                for x in range(cols)
            ]
            for y in range(lines)
        ]
        return {
            "material": TRAIL_MATERIAL,
            "like": TRAIL_LIKE,
            "tris": tris,
            "params": dict(RING_PARAMS),
            "render": dict(RING_RENDER),
            "texture": (0, TRAIL_PICTURE, rows),
            "anims": TRAIL_ANIMS,
        }

    def trail_poses(self, seq: Sequence, poses: list[Pose], step: tuple[int, float] = (0, 0.0)) -> None:
        """
        Put the trail's points into the poses of a clip of a swing (`seq`, played at `RATE`, its first
        piece the swing, its `window` its record's frames).

        On frame n a point some frames of his old is where his left hand was that long before, if a
        point of his game's trail was added then (`SwordTrail.frames`); a point older than the oldest
        added, or younger than the last, is put on that one, and before the first is added all are where
        the hand is: then the ribbon has no breadth. Where the swing carries him forward (`step`: so many
        frames at so many units a second), a point is put back by as far as he has gone since it was
        added, so that it stays where it was, as his game's do.
        """
        trail = self.sword_trail
        hand = PARTS.index("left_hand")
        added = trail.frames(seq.pieces[0].frames, seq.window[1])
        first, final = added[0], added[-1]
        span, count = trail.span(), len(self.slot_parts)

        def gone(frame: float) -> float:
            """How far the swing has carried him by frame `frame` of the clip, in units here."""
            return step[1] / 30.0 * min(max(frame, 0.0), step[0])

        seen = {}
        for n, pose in enumerate(poses):
            t = n / RATE
            for k, part in enumerate(self.slot_parts):
                when = t if t < first else min(t, final, max(first, t - span * k / (count - 1.0)))
                if when not in seen:
                    seen[when] = self.link.pose(seq.turns(when), seq.root(when))[hand]
                rot, at = seen[when]
                back = (gone(n) - gone(when * RATE)) / self.scale
                pose[part] = (rot, (at[0], at[1], at[2] - back))

    def _trail_shown(self, name: str, count: int) -> list[tuple[str, ...]]:
        """
        Mark a clip as one that shows the trail of his sword: its look shows on every frame, with the
        sword and shield out.

        Returns:
            What shows on each of its `count` + 1 frames.
        """
        if name not in self.trail["clips"]:
            self.trail["clips"].append(name)
        return [(*self.shown, TRAIL_LOOK)] * (count + 1)

    # -- his swings ----------------------------------------------------------------------------------

    def swing_moves(self) -> dict[str, Move]:
        """
        His swings, each played at `RATE` frames of this game to one of his.

        `hit` is the frame the game looks for something to hit on (the leaders' punch looks on one
        frame): the one nearest where the blade's tip is furthest ahead of him, of the frames his own
        game lets it hit on. `more` is the first frame of his coming back from the swing, from which
        his game, and the code here, take the next press.

        Returns:
            The swings, by the name of their clips.

        Raises:
            BuildError: If a swing does not come out as a hit and then a way back.
        """
        link, moves = self.link, {}
        hand, tip = link.blade()
        for name, kind in SWINGS:
            seq = link.swing(kind)
            first, last = seq.window

            def ahead(n: int, seq: Sequence = seq) -> float:
                rot, pos = link.pose(seq.turns(float(n)), seq.root(float(n)))[hand]
                return pos[2] + rotate(rot, tip)[2]

            furthest = max(range(first, last + 1), key=ahead)
            hit = int(furthest * RATE + 0.5)
            more = math.ceil(seq.pieces[0].frames * RATE)
            count = math.ceil((seq.frames - 1) * RATE)
            if not first * RATE <= hit <= last * RATE or not hit < more < count:
                raise BuildError(f"his {kind} swing does not come out as a hit and then a way back")
            # where it carries him: (for how many frames of the clip, at how many units a second)
            travel = seq.travel[1] * self.scale
            if abs(travel) > 0.01 * self.source_height * self.scale:
                frames = round((seq.pieces[0].frames - 1) * RATE)  # the frames the travel was taken out over
                step = (frames, travel * 30.0 / frames)
            elif kind.startswith("forward"):
                frames = round(LUNGE_FRAMES * RATE)
                step = (frames, LUNGE * self.scale * 30.0 / frames)
            else:
                step = (0, 0.0)
            if not 0 <= step[0] < more or not 0.0 <= step[1] < WALK:
                raise BuildError(f"his {kind} swing would carry him faster than he walks")
            times = [min(k / RATE, seq.frames - 1.0) for k in range(count + 1)]
            moves[name] = Move(
                "swing",
                kind,
                count,
                times,
                hit=hit,
                more=more,
                spark=("left_hand", hit),
                step=step,
                shows=self._trail_shown(name, count),
            )
        return moves

    # -- his roll ------------------------------------------------------------------------------------

    def dodge_moves(self) -> dict[str, Move]:
        """
        His roll, at the speed his own game plays it, longer than the game carries a dodger for, so
        that the game shows as much of it as it travels for. Nothing in it lands on anything.

        Returns:
            The roll, by the name of its clip.

        Raises:
            BuildError: If it does not come out as a clip that covers the dodge.
        """
        moves = {}
        for name, kind in DODGES:
            seq = self.link.keys(("roll" + ARMED,))
            per = RATE / self.link.roll_speed()
            count = math.ceil((seq.frames - 1) * per)
            times = [min(k / per, seq.frames - 1.0) for k in range(count + 1)]
            if not 19 <= count <= 250:
                raise BuildError(f"his roll does not come out as a clip that covers the dodge ({count + 1} frames)")
            moves[name] = Move("dodge", kind, count, times)
        return moves

    # -- a bomb held over his head -------------------------------------------------------------------

    def _upright(self, a: Animation, n: float) -> float:
        """How near upright his trunk is on a frame of an animation: 1 is upright."""
        waist, head = PARTS.index("waist"), PARTS.index("head")
        pose = self.link.pose(a.turns(float(n)), a.root(float(n)))
        v = [pose[head][1][k] - pose[waist][1][k] for k in range(3)]
        return v[1] / math.sqrt(sum(c * c for c in v))

    def bomb_keys(self, kind: str) -> Animation | Mixed:
        """What a clip of `BOMBS` is laid out from."""
        if kind == "lift":
            return Sequence(self.link, ("lift", "hold"), ease=BOMB_EASE, once=True)
        if kind == "put":
            return SetDown(self.link)
        a = self.link.animation("hold" if kind == "hold" else "throw" + ARMED)
        a.loops = kind == "hold"  # (the throw is played once: past its last frame, that frame)
        return a

    def lift_span(self) -> tuple[int, int]:
        """
        Where his lift is played from: the first frame on which his trunk is upright, having bent to
        the ground before it, to its last, at his own speed. (The animation is of lifting a thing off
        the ground; a bomb taken out needs none of the bending, and his legs, which go on walking,
        could not follow it.)

        Returns:
            (the frame of his lift its clip starts from, how many frames of the clip the rest of it takes).

        Raises:
            BuildError: If he is not upright for its last frames.
        """
        lift = self.link.animation("lift")
        bent = [n for n in range(lift.frames) if self._upright(lift, n) < UPRIGHT]
        # (the last frame he is bent on, among those before his hands go up: he leans back a little under the weight
        # later)
        low = [n for n in bent if n < lift.frames // 2]
        start = (max(low) + 1) if low else 0
        if not 0 <= start < lift.frames - 4 or self._upright(lift, start) < UPRIGHT:
            raise BuildError("his lift does not have him upright for its last frames")
        return start, math.ceil((lift.frames - 1 - start) * RATE)

    def put_gone(self) -> int:
        """
        The frame of his setting a thing down on which he lets go of it: the last on which his trunk
        is upright, his hands having come down in front of him, before he bows over it.

        Returns:
            That frame of his animation.

        Raises:
            BuildError: If it does not come out as his hands coming down, a bow and his standing again.
        """
        put = self.link.animation("put" + ARMED)
        bowed = [n for n in range(put.frames) if self._upright(put, n) < PUT_UPRIGHT]
        if (
            not bowed
            or not PUT_EASE < bowed[0] - 1 < put.frames - 1 - PUT_SETTLE
            or self._upright(put, put.frames - 1) < PUT_UPRIGHT
        ):
            raise BuildError(
                "his setting a thing down does not come out as his hands coming down, a bow and his standing again",
            )
        return bowed[0] - 1

    def bomb_moves(self) -> dict[str, Move]:
        """
        A bomb held over his head. `more` is the frame the code acts on: in the first clip his hands
        have arrived over his head (the bomb is made), in the throw and in the setting down they let
        go. The hold goes round over all of it, and takes its header from the leaders' clip for
        holding a Pikmin.

        Returns:
            The clips, by name.

        Raises:
            BuildError: If an animation does not come out as the clip it is for.
        """
        link, moves = self.link, {}
        lift, hold, throw = link.animation("lift"), link.animation("hold"), link.animation("throw" + ARMED)
        wrists, head = [PARTS.index("left_hand"), PARTS.index("right_hand")], PARTS.index("head")
        for name, kind in BOMBS:
            loop, like = None, "punch"
            if kind == "lift":
                top = lift.frames - 1.0  # the lift's last frame; the hold's frames follow one on
                start, rise = self.lift_span()
                count = rise + math.ceil((1.0 + BOMB_EASE) * RATE)
                times = [
                    min(start + k / RATE, top) if k <= rise else min(top + (k - rise) / RATE, top + 1.0 + BOMB_EASE)
                    for k in range(count + 1)
                ]

                def over(t: float) -> float:
                    """How far his lower wrist is over the joint of his head at a frame of his lift."""
                    pose = link.pose(lift.turns(t), lift.root(t))
                    return min(pose[w][1][1] - pose[head][1][1] for w in wrists)

                # the frame his hands arrive: the first on which they are `ARRIVED` of the way over his head that
                # the lift leaves them
                arrived = [k for k in range(rise + 1) if over(times[k]) >= ARRIVED * over(top)]
                if over(top) <= 0 or over(float(start)) >= 0 or not arrived or not 0 < arrived[0] <= rise:
                    raise BuildError("his lift does not bring his hands from below his head to over it")
                more = arrived[0]
            elif kind == "hold":
                round_ = hold.frames - 1.0
                if apart(hold.turns(round_), hold.turns(0.0)) > 2.0:
                    raise BuildError("his hold does not come round: its last frame is not its first")
                count = math.ceil(round_ * RATE)
                # (the last frame of the clip is exactly the first: the whole of his round, whatever `RATE` leaves over)
                times = [(BOMB_EASE + k * round_ / count) % round_ for k in range(count)] + [float(BOMB_EASE)]
                more, loop, like = None, (0, count), HOLD_LIKE
            elif kind == "put":
                put = link.animation("put" + ARMED)
                gone = self.put_gone()
                rate = RATE / PUT_SPEED  # frames of the clip to one of his
                count = math.ceil((put.frames - 1) * rate)
                times = [min(k / rate, put.frames - 1.0) for k in range(count + 1)]
                more = int(gone * rate + 0.5)
                if not 0 < more < count - 2:
                    raise BuildError("his setting a thing down does not come out as a clip to let go in")
            else:

                def up(n: int) -> bool:
                    """Whether both his wrists are above the joint of his head on a frame of the throw."""
                    pose = link.pose(throw.turns(float(n)), throw.root(float(n)))
                    return all(pose[w][1][1] > pose[head][1][1] for w in wrists)

                down = [n for n in range(throw.frames) if not up(n)]
                gone = (down[0] if down else throw.frames) - 1
                count = math.ceil((throw.frames - 1) * RATE)
                times = [min(k / RATE, throw.frames - 1.0) for k in range(count + 1)]
                more = int(gone * RATE + 0.5)
                if not 0 < gone < throw.frames // 2 or not 0 < more < count:
                    raise BuildError("his throw does not come out as a swing over his head and a way back")
            if not 8 <= count <= 250 or any(t < 0 for t in times):
                raise BuildError(f"his {kind} does not come out as a clip ({count + 1} frames)")
            moves[name] = Move("bomb", kind, count, times, more=more, loop=loop, like=like)
            moves[name].shows = self.bomb_shows(moves[name])
        return moves

    def bomb_shows(self, move: Move) -> list[tuple[str, ...]]:
        """
        What shows on each frame of a clip of `BOMBS`. The sword and shield are on his back and his
        hands open from the clip's first frame; in the throw and the setting down they are in his
        hands again from the first frame after which his arms are never more than `BACK` degrees from
        where the clip leaves them (his stand with them out) and his trunk is upright.

        Returns:
            The looks that show, on each frame.

        Raises:
            BuildError: If his arms are not back in his stand after he lets go.
        """
        away, drawn = self.sets["away"], self.sets["drawn"]
        count = move.count + 1
        if move.kind in {"lift", "hold"}:
            return [away] * count
        poses = self.move_poses(move)
        arms = [PARTS.index(name) for name in ARMS]
        waist, head = PARTS.index("waist"), PARTS.index("head")
        off = [apart({i: pose[i][0] for i in arms}, {i: poses[-1][i][0] for i in arms}) for pose in poses]

        def upright(pose: Pose) -> float:
            v = [pose[head][1][k] - pose[waist][1][k] for k in range(3)]
            return v[1] / math.sqrt(sum(c * c for c in v))

        # (and he is standing up again: setting a bomb down, his arms are back while he is still bowed over it)
        back = max(k for k in range(count) if off[k] > BACK or upright(poses[k]) < PUT_UPRIGHT) + 1
        # (and never before the bomb has left his hands)
        if not move.more < back < count:
            raise BuildError(f"his {move.kind} does not bring his arms back to his stand after he lets go")
        return [away] * back + [drawn] * (count - back)

    # -- what he does left standing ------------------------------------------------------------------

    def idle_keys(self, kind: str) -> Mixed:
        """What a clip of `IDLES` is laid out from."""
        return Sit(self.link) if kind == "sit" else Fidget(self.link, kind)

    def idle_moves(self) -> dict[str, Move]:
        """
        What he does when left standing. His sitting down goes round between two frames: the game
        goes back from the second to the first until it is told to finish, as it does in the
        leaders' own sitting clips.

        Returns:
            The clips, by name.

        Raises:
            BuildError: If one does not come out as a clip.
        """
        moves = {}
        for name, kind in IDLES:
            seq = self.idle_keys(kind)
            if kind == "sit":
                down, stay, up = (math.ceil(n * RATE) for n in (seq.down, seq.round, seq.up))
                times = (
                    [min(k / RATE, seq.down) for k in range(down + 1)]
                    + [seq.down + j * seq.round / stay for j in range(1, stay + 1)]
                    + [seq.down + seq.round + min(i / RATE, seq.up) for i in range(1, up + 1)]
                )
                loop = (down, down + stay)
            else:
                count = math.ceil(seq.last * RATE)
                times = [min(k / RATE, seq.last) for k in range(count + 1)]
                loop = None
            if not 30 <= len(times) - 1 <= 400 or any(b < a for a, b in pairwise(times)):
                raise BuildError(f"his {kind} does not come out as a clip ({len(times)} frames)")
            moves[name] = Move("idle", kind, len(times) - 1, times, loop=loop)
        return moves

    # -- his bow -------------------------------------------------------------------------------------

    def _fists(self, pose: Pose) -> list[Vector]:
        """Where the middles of his fists are in a pose: the left (the string hand's), then the right (the bow's)."""
        out = []
        for limb, middle in zip(("left_hand", "right_hand"), self.fists, strict=True):
            rot, pos = pose[PARTS.index(limb)]
            moved = rotate(rot, middle)
            out.append(tuple(pos[k] + moved[k] for k in range(3)))
        return out

    def bow_turn(self) -> float:
        """
        How far his upper body is turned in the bow's clips, in radians about the upright line
        through his hips: what brings the line from his string hand to his bow hand, in the first
        frame of his aim, to dead ahead. His game turns his trunk to where he aims.

        Returns:
            The angle.

        Raises:
            BuildError: If in his aim the bow is not held out ahead of the string hand.
        """
        aim = BowKeys(self.link, "aim")
        left, right = self._fists(self.link.pose(aim.turns(0.0), aim.root(0.0)))
        across, along = right[0] - left[0], right[2] - left[2]
        if along <= abs(across):
            raise BuildError("in his aim the bow is not held out ahead of the string hand")
        return math.atan2(-across, along)

    @staticmethod
    def _turned(pose: Pose, angle: float) -> Pose:
        """A pose with every part above his legs turned by `angle` about the upright line through his root."""
        turn = euler_matrix(0.0, angle, 0.0)
        pivot = pose[0][1]
        out = list(pose)
        for i, name in enumerate(PARTS):
            if STOCK.get(name) in UPPER_BONES:
                rot, pos = pose[i]
                moved = rotate(turn, tuple(pos[k] - pivot[k] for k in range(3)))
                out[i] = (matmul(turn, rot), tuple(pivot[k] + moved[k] for k in range(3)))
        return out

    def _arrow(self, tail: Vector, way: Vector) -> tuple[Matrix, Vector]:
        """The arrow's part with its nock at `tail`, pointing along `way` (a unit vector), feathers upright."""
        sign = 1.0 if self.arrow.tip > self.arrow.tail else -1.0
        ex = tuple(sign * c for c in way)
        up = (0.0, 1.0, 0.0)
        lean = sum(u * e for u, e in zip(up, ex, strict=True))
        ey = tuple(u - lean * e for u, e in zip(up, ex, strict=True))
        size = math.sqrt(sum(c * c for c in ey))
        ey = tuple(c / size for c in ey)
        ez = (ex[1] * ey[2] - ex[2] * ey[1], ex[2] * ey[0] - ex[0] * ey[2], ex[0] * ey[1] - ex[1] * ey[0])
        rot = [[ex[i], ey[i], ez[i]] for i in range(3)]
        return rot, tuple(tail[k] - self.arrow.tail * ex[k] for k in range(3))

    def bow_flight(self) -> tuple[Vector, float]:
        """
        The arrow's flight, in this game's units and from where he stands: it flies dead ahead,
        `ARROW_SPEED` a frame for `ARROW_FRAMES` frames, its nock coming down from the height of his
        string hand to `ARROW_LANDS`.

        Returns:
            Where its nock is as it leaves the string (across, up, ahead), and how far it sinks each frame.
        """
        aim = BowKeys(self.link, "aim")
        left, _ = self._fists(self._turned(self.link.pose(aim.turns(0.0), aim.root(0.0)), self.bow_turn()))
        start = self.place(left)
        return start, (start[1] - ARROW_LANDS) / float(ARROW_FRAMES)

    def bow_moves(self) -> dict[str, Move]:
        """
        His bow. The aim goes round, and takes its header from the leaders' clip for holding a
        Pikmin; the loosing is held while the arrow flies, one frame longer than the flight. In each
        the bow shows, with the sword and shield on his back, and the arrow where it is to be seen:
        from a few frames into raising the bow, from where his hand has come forward for the string
        as he takes the next arrow, and while it flies.

        Returns:
            The clips, by name.

        Raises:
            BuildError: If one does not come out as a clip.
        """
        moves = {}
        bow = self.sets["bow"]
        for name, kind in BOWS:
            seq = BowKeys(self.link, kind)
            loop, like = None, "punch"
            if kind == "aim":
                count = math.ceil(seq.last * RATE)
                times = [k * seq.last / count for k in range(count)] + [0.0]
                loop, like = (0, count), HOLD_LIKE
                arrow = ["nocked"] * (count + 1)
            elif kind == "loose":
                count = max(math.ceil(seq.last * RATE), ARROW_FRAMES + 1)
                times = [min(k / RATE, seq.last) for k in range(count + 1)]
                arrow = ["flying" if k <= ARROW_FRAMES else None for k in range(count + 1)]
            else:
                count = math.ceil(seq.last * RATE)
                times = [min(k / RATE, seq.last) for k in range(count + 1)]
                if kind == "ready":
                    arrow = [None if k < ARROW_SHOWS else "nocked" for k in range(count + 1)]
                else:
                    # the next arrow is in his hand from the frame that hand is furthest forward: he has reached
                    # the string
                    left = PARTS.index("left_hand")
                    ahead = [self.link.pose(seq.turns(t), seq.root(t))[left][1][2] for t in times]
                    reached = ahead.index(max(ahead))
                    if not 0 < reached < count - 2:
                        raise BuildError("taking the next arrow, his hand does not come forward and draw back")
                    arrow = [None if k < reached else "nocked" for k in range(count + 1)]
            if not 8 <= count <= 250:
                raise BuildError(f"his {kind} with the bow does not come out as a clip ({count + 1} frames)")
            shows = [bow + (("arrow",) if a else ()) for a in arrow]
            moves[name] = Move("bow", kind, count, times, loop=loop, like=like, shows=shows, arrow=arrow)
        return moves

    def bow_poses(self, move: Move) -> list[Pose]:
        """His pose at each frame of a clip of `BOWS`, turned to where he aims, the arrow's part where the arrow is."""
        link = self.link
        seq, angle = BowKeys(link, move.kind), self.bow_turn()
        _, sink = self.bow_flight()
        aim = BowKeys(link, "aim")
        string = self._fists(self._turned(link.pose(aim.turns(0.0), aim.root(0.0)), angle))[0]
        size = math.sqrt(ARROW_SPEED**2 + sink**2)
        way = (0.0, -sink / size, ARROW_SPEED / size)
        poses = []
        for k, (t, arrow) in enumerate(zip(move.times, move.arrow, strict=True)):
            pose = self._turned(link.pose(seq.turns(t), seq.root(t)), angle * seq.twist(t))
            if arrow == "flying":
                # (in his model's units: from the string as it was in his aim, dead ahead and sinking)
                tail = (string[0], string[1] - k * sink / self.scale, string[2] + k * ARROW_SPEED / self.scale)
                pose[self.nock] = self._arrow(tail, way if k else (0.0, 0.0, 1.0))
            else:
                # nocked, or hidden (and then it is where it would be nocked): from the string hand's fist to the
                # bow hand's
                left, right = self._fists(pose)
                reach = math.sqrt(sum((right[i] - left[i]) ** 2 for i in range(3)))
                pose[self.nock] = self._arrow(left, tuple((right[i] - left[i]) / reach for i in range(3)))
            poses.append(pose)
        return poses

    # -- a charge of his sword -----------------------------------------------------------------------

    def spin_keys(self, kind: str) -> Mixed:
        """What a clip of `SPINS` is laid out from: `Charge`, his spin as a swing, or his way out of the stance."""
        if kind == "charge":
            return Charge(self.link)
        if kind == "spin":
            return self.link.swing("spin")
        return Sequence(self.link, (self.link.spins()["charge_out"],), once=True)

    def spin_moves(self) -> dict[str, Move]:
        """
        A charge of his sword and its spin, each at `RATE`. The charge goes round between two frames
        inside it. On every frame of the spin the blade can hit on (`window`), the spark's bone is
        put on the blade; and the spin gives the ring's looks their sizes. None has a marker: the
        code counts frames.

        Returns:
            The clips, by name.

        Raises:
            BuildError: If one does not come out as a clip.
        """
        moves = {}
        for name, kind in SPINS:
            seq = self.spin_keys(kind)
            loop, window, spark, sizes, shows = None, None, None, None, None
            if kind == "charge":
                count = round(seq.frames * RATE)
                going = seq.going * RATE
                if abs(count - seq.frames * RATE) > 1e-6 or abs(going - round(going)) > 1e-6:
                    raise BuildError("his charge's stance does not go round on whole frames here")
                times = [k / RATE for k in range(count + 1)]
                loop = (round(going), count)
            else:
                count = math.ceil((seq.frames - 1) * RATE)
                times = [min(k / RATE, seq.frames - 1.0) for k in range(count + 1)]
                if kind == "spin":
                    first, last = seq.window
                    window = (math.ceil(first * RATE), math.floor(last * RATE))
                    spark = ("left_hand", window)
                    sizes = self.ring_sizes(count)
                    shows = self._trail_shown(name, count)
            if not 15 <= count <= 250:
                raise BuildError(f"his {kind} does not come out as a clip ({count + 1} frames)")
            moves[name] = Move(
                "spin",
                kind,
                count,
                times,
                loop=loop,
                window=window,
                spark=spark,
                sizes=sizes,
                shows=shows,
            )
        return moves

    def spin_sweep(self) -> list[tuple[float, float]]:
        """
        Where the blade is on each frame of his spin on which it can hit, from the clip's own poses.

        Returns:
            [(the way to it from where he stands, in radians from straight ahead towards his left;
            how far out its tip is, in units here)].
        """
        move = self.moves[SPINS[1][0]]
        hand, tip = self.link.blade()
        out = []
        for pose in self.move_poses(move)[move.window[0] : move.window[1] + 1]:
            rot, pos = pose[hand]
            moved = rotate(rot, tip)
            x, z = (pos[k] + moved[k] for k in (0, 2))
            out.append((math.atan2(x, z), math.hypot(x, z) * self.scale))
        return out

    # -- what the builder asks of a clip -------------------------------------------------------------

    def keys(self, move: Move) -> Animation | Mixed | Keys:
        """What a move's clip is laid out from."""
        if move.family == "swing":
            return self.link.swing(move.kind)
        if move.family == "dodge":
            return self.link.keys(("roll" + ARMED,))
        if move.family == "bomb":
            return self.bomb_keys(move.kind)
        if move.family == "idle":
            return self.idle_keys(move.kind)
        return self.spin_keys(move.kind)

    def move_poses(self, move: Move) -> list[Pose]:
        """His pose at each frame of a move's clip."""
        key = (move.family, move.kind, move.count)
        if key not in self._poses:
            if move.family == "bow":
                poses = self.bow_poses(move)
            else:
                seq = self.keys(move)
                poses = [self.link.pose(seq.turns(t), seq.root(t)) for t in move.times]
                if move.family == "spin" and move.kind == "spin":
                    # The ring is made where his game keeps a point of him as the spin begins (his waist), and lies flat
                    # there: its part, turned as `ring_turn` says, at his waist on the spin's first frame.
                    waist = poses[0][PARTS.index("waist")][1]
                    for n, pose in enumerate(poses):
                        pose[self.ring_part] = (self.ring_turn(n), waist)
                    self.trail_poses(seq, poses)
                elif move.family == "swing":
                    self.trail_poses(seq, poses, move.step)
            self._poses[key] = poses
        return self._poses[key]

    def move_point(self, pose: Pose) -> Vector:
        """Where the spark of a hit is drawn: on the blade, `BLADE` of the way from his hand to its tip."""
        hand, tip = self.link.blade()
        rot, pos = pose[hand]
        along = rotate(rot, tuple(c * BLADE for c in tip))
        return self.place(tuple(pos[k] + along[k] for k in range(3)))


def _middle(tris: list) -> Vector:
    """The middle of a list's corners, in the space of the limb it was read for."""
    points = sorted({c[1] for _, corners, _ in tris for c in corners})
    return tuple(sum(p[k] for p in points) / float(len(points)) for k in range(3))


# -- where a bomb sits in his hands ------------------------------------------------------------------


def bomb_at(link: Link, pose: Pose, scale: float, radius: float, floor: float | None = None) -> tuple[float, float]:
    """
    Where a ball of `radius` (in this game's units) is when he has it in his hands in a pose.

    The ball is on his centre line, out beyond his hands along the line his arms lie on (from
    between his shoulders to between his wrists): its surface is `HAND_CLEAR` past the furthest the
    shapes of his hands reach along that line. Over his head that puts it above his fingertips, and
    with his arms down in front of him, in front of them. With `floor`, its lowest point is no lower
    than that, and where that raises it, it is as much further ahead as keeps it as far from his
    wrists. `link` should have open hands. `scale` is this game's units to one of his model's.

    Returns:
        (units ahead of him, units above his feet) for the ball's lowest point.
    """

    def at(a: str, b: str) -> list[float]:
        return [(pose[PARTS.index(a)][1][k] + pose[PARTS.index(b)][1][k]) / 2.0 for k in range(3)]

    shoulders, wrists = at("left_upperarm", "right_upperarm"), at("left_hand", "right_hand")
    arm = [0.0, wrists[1] - shoulders[1], wrists[2] - shoulders[2]]
    length = math.sqrt(sum(c * c for c in arm))
    arm = [c / length for c in arm]
    hands = [p for _, name, tri in link.triangles(pose) if name in {"left_hand", "right_hand"} for p, _, _ in tri]
    tip = max(sum((p[k] - wrists[k]) * arm[k] for k in range(3)) for p in hands)
    reach = tip * scale + HAND_CLEAR + radius
    ahead, lowest = wrists[2] * scale + arm[2] * reach, wrists[1] * scale + arm[1] * reach - radius
    if floor is not None and lowest < floor:
        lowest = float(floor)
        ahead = wrists[2] * scale + math.sqrt(max(reach * reach - (lowest + radius - wrists[1] * scale) ** 2, 0.0))
    return ahead, lowest


def bomb_seat(link: Link, scale: float, radius: float) -> tuple[float, float]:
    """Where a ball of `radius` sits while he holds it over his head, averaged over his hold (`bomb_at`)."""
    hold = link.animation("hold")
    seats = [
        bomb_at(link, link.pose(hold.turns(float(n)), hold.root(float(n))), scale, radius) for n in range(hold.frames)
    ]
    return sum(s[0] for s in seats) / len(seats), sum(s[1] for s in seats) / len(seats)


def bomb_put_path(link: Link, scale: float, radius: float, frames: int) -> list[tuple[float, float]]:
    """
    Where the ball is on each frame of the clip of his setting it down, from the frame after the
    clip's first to the one he lets go on (`frames` of them), by `bomb_at`, and no lower than
    `PUT_DROP`. His trunk and arms are where his animation has them, his root with them: the clip is
    played on his upper body, whose bones are placed by where his animation has each part.

    Returns:
        (units ahead of him, units above his feet) for the ball's lowest point, on each of those frames.
    """
    seq, rate = SetDown(link), RATE / PUT_SPEED
    return [
        bomb_at(link, link.pose(seq.turns(k / rate), seq.root(k / rate)), scale, radius, floor=PUT_DROP)
        for k in range(1, frames + 1)
    ]
