; His swings, a swing at nothing, and his roll.
;
; The leaders' punch state plays one clip and ends with it. For Link, entering it plays a swing of
; his sword, and a press of A once a clip has passed its "the next press is taken from here" marker
; plays the next swing and keeps the state alive. Each frame he is told where to go, which the punch
; state never does: to stop, or, for the first frames of a swing that steps forward in his own game,
; to go the way he faces. The game still does the hitting; `s_reach`, `s_furthest` and `s_strength`
; hand it his numbers in place of a leader's.
;
; Names in capitals and the game's own functions come from moves.py. Every other leader takes the
; first way out of each routine, with the answer the game's own word would have given.

; ==== entering the punch state: r0 the animation, r2 which part of the body (0 if he is moving, 2 if not), r4 the state ====
.org ROOM_SWORD
s_enter_a:
    ldr   r12, [r4, #0x5c]          ; the leader
    b     s_enter
s_enter_b:
    ldr   r12, [r4, #0x28]          ; the same, from the twin's side of the state
s_enter:
    mov   r1, #0x42                 ; the leaders' punch, as the game had it
    ldr   r12, [r12, #0x81c]
    cmp   r12, #LINK                ; only the leader whose model is his
    bxne  lr
    ldr   r12, bytes_from_enter
s_enter_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #ASKED]
    cmp   r1, #0
    bne   b_enter                   ; a bomb was asked for: this is not a swing
    mov   r1, #0
    strh  r1, [r12]                 ; the first swing of a run; no press being taken
    strb  r1, [r12, #FRAMES]        ; its first frame
    cmp   r2, #0                    ; the game's own test, just made: is he moving?
    moveq r1, #1                    ; moving, which is the stick pushed: the swing straight down
    strb  r1, [r12, #KIND]          ; standing: the swing from his right
    add   r1, r1, r1, lsl #1        ; (a kind's clips are together, RUN of them)
    add   r1, r1, #SWING_CLIP
    mov   r2, #2                    ; on the whole body, moving or not
    bx    lr
bytes_from_enter: .word STATE - (s_enter_at + 8)

; ==== each frame of the punch state, "has the clip ended?": r0 the animation, r4 the state, r7 the leader ====
s_update:
    push  {r4, r5, r6, lr}
    bl    anim_flags
    mov   r5, r0                    ; the answer the state asked for
    ldr   r0, [r7, #0x81c]
    cmp   r0, #LINK
    bne   s_done
    ldr   r6, bytes_from_update
s_update_at:
    add   r6, pc, r6                ; r6 -> the bytes
    ldrb  r0, [r6, #MODE]
    cmp   r0, #0
    bne   b_update                  ; he has a bomb out: this is not a swing
    ; ---- where he goes this frame: forward for the frames this swing steps, and otherwise he stands ----
    ldrb  r0, [r6, #KIND]
    ldrb  r1, [r6]
    add   r0, r0, r0, lsl #1
    add   r0, r0, r1                ; which clip, counted from his first
    ldrb  r1, [r6, #FRAMES]         ; frames into it
    add   r3, pc, r0, lsl #3        ; (two words a clip, straight after the branch below)
    ldr   r2, [r3, #8]              ; for how many frames it steps
    ldr   r4, [r3, #12]             ; and how fast, as a share of a leader's walk
    b     s_step
    .words SWING_STEPS
s_step:
    cmp   r1, #255
    addcc r3, r1, #1
    strbcc r3, [r6, #FRAMES]        ; one frame further into the swing
    cmp   r1, r2
    bcs   s_stand
    mov   r0, r7
    bl    facing_of
    sub   sp, sp, #16
    ldr   r1, [r0, #0x3c]           ; the way he faces: across ...
    str   r1, [sp]
    mov   r1, #0
    str   r1, [sp, #4]              ; ... level ...
    ldr   r1, [r0, #0x5c]           ; ... and along
    str   r1, [sp, #8]
    mov   r12, r4
    vmov  s0, r12                   ; the share of his speed
    mov   r0, r7
    mov   r1, sp
    ldr   r12, [r7]
    ldr   r12, [r12, #0x170]
    blx   r12                       ; go that way: the call the walk state moves him by
    add   sp, sp, #16
    b     s_moved
s_stand:
    mov   r0, r7
    ldr   r12, [r7]
    ldr   r12, [r12, #0x19c]
    blx   r12                       ; stop: the call the walk state makes with the stick let go
s_moved:
    tst   r5, #0x800                ; the clip's "the next press is taken from here" marker
    movne r0, #1
    strbne r0, [r6, #TAKING]
    ldrb  r0, [r6, #TAKING]
    cmp   r0, #0
    beq   s_done                    ; not yet
    mov   r0, r7
    bl    player_of
    cmn   r0, #1
    beq   s_done                    ; nobody is controlling him
    mov   r4, r0
    ldr   r0, mode_from_here        ; has A just been pressed? Read as the walk state reads it
s_mode_at:
    ldr   r0, [pc, r0]
    cmp   r0, #0
    beq   s_pad
    bl    shared_mode
    cmp   r0, #0
    beq   s_pad
    ldr   r0, game_from_here
s_game_at:
    ldr   r0, [pc, r0]
    bl    shared_pad
    mov   r4, r0
s_pad:
    ldr   r1, pads_from_here
s_pads_at:
    ldr   r1, [pc, r1]
    add   r1, r1, #0x10
    cmp   r4, #4
    addcc r1, r1, r4, lsl #10       ; + player * 0x4f0
    addcc r1, r1, r4, lsl #8
    subcc r1, r1, r4, lsl #4
    ldrb  r0, [r1, #8]              ; what was pressed this frame
    tst   r0, #1                    ; A
    beq   s_done
    ldr   r0, [r1, #0x210]          ; the stick, across and along
    ldr   r2, [r1, #0x214]
    bic   r0, r0, #0x80000000
    bic   r2, r2, #0x80000000
    cmp   r0, r2
    movcc r0, r2                    ; how far over it is, either way (as bits: neither is below zero now)
    ldr   r2, pushed
    cmp   r0, r2
    movhi r0, #1                    ; pushed: the swing straight down
    movls r0, #0                    ; let go: the swing from his right
    ldrb  r2, [r6, #KIND]           ; the kind of the swing he is in
    ldrb  r1, [r6]                  ; and which of its run that is
    cmp   r0, r2
    addeq r1, r1, #1                ; the same kind again: the next of the run
    movne r1, #0                    ; the other kind: a run of it begins
    cmp   r1, #RUN
    movcs r1, #0                    ; after the later blow, the first again
    mov   r2, #0
    strh  r1, [r6]                  ; which swing this is; no press is being taken
    strb  r0, [r6, #KIND]
    strb  r2, [r6, #FRAMES]         ; its first frame
    add   r0, r0, r0, lsl #1
    add   r1, r1, r0
    add   r1, r1, #SWING_CLIP       ; its clip
    ldr   r0, [r7, #0x374]          ; the leader's animation
    ldr   r12, [r0]
    mov   r2, #2                    ; on the whole body
    ldr   r12, [r12, #0xc4]
    blx   r12                       ; play it
    bic   r5, r5, #0x80000000       ; and the state does not end on this frame
s_done:
    mov   r0, r5
    pop   {r4, r5, r6, pc}
bytes_from_update: .word STATE - (s_update_at + 8)
mode_from_here:    .word mode_object - (s_mode_at + 8)
game_from_here:    .word game_object - (s_game_at + 8)
pads_from_here:    .word pads - (s_pads_at + 8)
pushed:            .word PUSHED

; ==== what a hit reaches and how hard it lands: r7 the leader. Each of these stands where the
; ==== game set one number, and hands back the game's own number for every other leader ====
s_reach:                            ; r0: the radius searched; s18 (already 4.0): how far ahead of him
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    ldr   r0, radius_theirs
    bxne  lr
    ldr   r0, radius_his
    ldr   r12, ahead_his
    vmov  s18, r12
    bx    lr
s_furthest:                         ; s16: the furthest from him a hit may land; r0 is the game's, kept
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    ldr   r12, furthest_theirs
    ldreq r12, furthest_his
    vmov  s16, r12
    bx    lr
s_strength:                         ; r0: the strength of the hit
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    ldr   r0, power_theirs
    ldreq r0, power_his
    bx    lr
radius_theirs:   .word RADIUS_THEIRS
radius_his:      .word RADIUS
ahead_his:       .word AHEAD
furthest_theirs: .word FURTHEST_THEIRS
furthest_his:    .word FURTHEST
power_theirs:    .word POWER_THEIRS
power_his:       .word POWER

; ==== the walk state, where a press of A has found nothing to do: r0 -> the game's settings for a
; ==== punch at nothing, r4 the state. Hands back whether to punch; r1-r3 and r12 are the caller's to lose ====
s_air:
    ldrb  r0, [r0, #0x51]           ; the game's own switch (the instruction this call stands in place of)
    cmp   r0, #0
    bxne  lr                        ; on: as the game has it, for everyone
    ldr   r12, [r4, #0x5c]          ; the leader
    ldr   r1, [r12, #0x81c]
    cmp   r1, #LINK
    bxne  lr                        ; anyone else: off, as it was
    push  {r4, lr}
    mov   r0, r12
    bl    party_of                  ; those who follow him, himself among them (or nothing)
    cmp   r0, #0
    ldrne r0, [r0, #0x68]           ; how many
    cmp   r0, #1
    moveq r0, #1                    ; nobody but himself: there is nothing to throw, so A swings his sword
    movne r0, #0
    pop   {r4, pc}

; ==== entering the dodge, for a leader or a Pikmin alike: r0 the animation, r1 the clip the game
; ==== chose, r3 its own function that plays a clip, which it calls next, r4 the state ====
s_dodge:
    mov   r2, #2                    ; on the whole body (the instruction this call stands in place of)
    ldr   r12, [r4, #0x5c]          ; whoever is dodging
    ldr   r12, [r12, #0x81c]
    cmp   r12, #LINK                ; only the leader whose model is his
    bxne  lr
    cmp   r1, #THEIR_ROLL           ; the game turns him the way he goes and rolls him forward
    moveq r1, #ROLL_CLIP            ; his roll
    bx    lr
