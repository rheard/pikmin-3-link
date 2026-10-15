; What he does when he is left standing.
;
; Left standing with nobody playing him, a leader now and then plays one of the Pikmin's clips once,
; and in time sits down and lies down to sleep. Link plays the next of five things of his own
; instead, in turn; he sits with a clip of his own (its number is set in hooks.s) and never lies down.

; ==== whoever is left standing, leader or Pikmin: the game has chosen a clip for it to play once, and is about
; ==== to start that. r0 the behaviour, r1 = 1, r2 what is asked for (the clip's number 4 into it), r5 whoever it is ====
.org ROOM_IDLE
i_fidget:
    ldr   r12, [r5, #0x81c]
    cmp   r12, #LINK                ; only the leader whose model is his
    bne   i_asked
    ldr   r12, [r2, #4]
    cmp   r12, #THEIR_STAND         ; the stand every Pikmin shares, in which his own bones rest:
    moveq r12, #HIS_STAND           ; ... the leaders' own entry for standing, which is his stand
    beq   i_put
    ldr   r3, bytes_from_idle       ; anything else: the next of the things he does, in turn
i_fidget_at:
    add   r3, pc, r3
    ldrb  r12, [r3, #IDLE]
    add   r12, r12, #1
    cmp   r12, #TURNS
    movcs r12, #0
    strb  r12, [r3, #IDLE]
    add   r12, r12, #IDLE_CLIP
i_put:
    str   r12, [r2, #4]
i_asked:
    b     make_once                 ; as the game was about to (it comes back to the game's own code)
bytes_from_idle: .word STATE - (i_fidget_at + 8)

; ==== the one sitting is to lie down and sleep: r0 whoever it is, r4 the rest ====
i_sleep:
    ldr   r1, [r0, #0x81c]
    ldr   r0, [r0, #0x374]          ; its animation, as the game had it (the instruction this call stands in place of)
    cmp   r1, #LINK
    bxne  lr                        ; anyone else lies down
    b     rest_waits                ; he stays sitting: the rest's wait begins again, as it does after the game's own choice
