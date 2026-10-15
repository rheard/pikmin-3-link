; A bomb held over his head: the part of it in the punch state, straight after swing.s.
;
; D-pad Down puts him in the punch state with a note that a bomb was asked for (bombs.s), and the
; state's two calls to the sword's code turn aside for it. Entering, his legs are left following his
; speed, as the game has them while he holds a Pikmin, and his upper body plays his lift. On the
; frame his hands arrive a bomb is made over his head and hung on him; he holds it and walks where the
; stick says until A throws it or D-pad Down sets it down (put.s). The punch state lasts as long as
; "has the clip ended?" is answered no, and it reads no button itself, so this code answers for it.

; ==== entering the punch state with a bomb asked for: r0 the animation, r3 the game's function that
; ==== plays a clip, which it calls next with r1 the clip and r2 the part of him, r12 the bytes ====
b_enter:
    push  {r0, r3, r4, lr}
    mov   r1, #0
    strb  r1, [r12, #ASKED]
    mov   r1, #RAISING
    strb  r1, [r12, #MODE]          ; he is raising his hands
    mov   r1, #1
    mov   r2, #1
    bl    layer                     ; his legs go on following his speed, as when he has a Pikmin in his hand
    mov   r1, #LIFT_CLIP
    mov   r2, #0                    ; and his upper body lifts
    pop   {r0, r3, r4, pc}

; ==== each frame of the punch state with a bomb out: r4 the state, r5 what his animation passed this
; ==== frame (below zero: its clip has ended), r6 the bytes, r7 the leader. The sword's code answers ====
b_update:
    ldrb  r0, [r6, #MODE]
    cmp   r0, #PUTTING
    beq   b_putting                 ; setting it down, he stands where he is
    mov   r0, r4
    mov   r1, r7
    mov   r2, #0
    bl    steer                     ; he goes where the stick says, as in the walk state
b_steered:
    bl    b_light                   ; (one just let go is knocked on, and lit when its frames are up)
    ldrb  r0, [r6, #MODE]
    cmp   r0, #HOLDING
    beq   b_holding
    bhi   b_throwing                ; (throwing it or setting it down: either way his hands let go on the clip's marker)
    tst   r5, #0x800                ; raising his hands: the frame they arrive
    beq   b_raised
    bl    b_make                    ; a bomb, over his head and hung on him
    cmp   r0, #0
    beq   b_over                    ; the map has none to give: his hands come down
b_raised:
    cmp   r5, #0
    bge   b_stay
    bl    b_ours                    ; the clip has ended: he holds it
    cmp   r0, #0
    beq   b_over                    ; (nothing came of it)
    mov   r0, #HOLDING
    mov   r1, #HOLD_CLIP
    b     b_play
b_holding:
    bl    b_ours
    cmp   r0, #0
    beq   b_over                    ; it is no longer on him: his hands come down
    mov   r0, r7
    bl    b_pad                     ; the pad of whoever plays him
    cmp   r0, #0
    beq   b_go                      ; nobody does: he throws it
    ldrb  r1, [r0, #8]              ; what was pressed this frame
    ldr   r2, [r0, #0x238]          ; and on the controller behind the pad
    cmp   r2, #0
    ldrbne r2, [r2, #0xa]
    tst   r2, #4                    ; down on the D-pad again: he sets it down
    bne   b_put
    tst   r1, #1                    ; A: he throws it
    beq   b_stay
b_go:
    mov   r0, #THROWING
    mov   r1, #THROW_CLIP
    b     b_play
b_throwing:
    tst   r5, #0x800                ; the frame his hands let go
    beq   b_swung
    bl    b_ours
    cmp   r0, #0
    blne  b_fly
b_swung:
    cmp   r5, #0
    bge   b_stay
b_over:
    bl    b_drop                    ; no bomb is out now (and one still on him, which there should not be, goes)
    orr   r5, r5, #0x80000000       ; the state ends, as after a swing
    b     s_done
b_play:                             ; r0: what he does next; r1: its clip
    strb  r0, [r6, #MODE]
    ldr   r0, [r7, #0x374]          ; his animation
    ldr   r12, [r0]
    mov   r2, #0                    ; on his upper body
    ldr   r12, [r12, #0xc4]
    blx   r12
b_stay:
    bic   r5, r5, #0x80000000       ; the state goes on
    b     s_done

; ---- the pad of whoever plays him, found as the walk state finds it: r0 the leader; r0 the pad, or 0 ----
b_pad:
    push  {r4, lr}
    bl    player_of
    cmn   r0, #1
    moveq r0, #0
    popeq {r4, pc}                  ; nobody is
    mov   r4, r0
    ldr   r0, mode_from_pad
b_pad_at1:
    ldr   r0, [pc, r0]
    cmp   r0, #0
    beq   b_pad_have
    bl    shared_mode
    cmp   r0, #0
    beq   b_pad_have
    ldr   r0, game_from_pad
b_pad_at2:
    ldr   r0, [pc, r0]
    bl    shared_pad
    mov   r4, r0
b_pad_have:
    ldr   r0, pads_from_pad
b_pad_at3:
    ldr   r0, [pc, r0]
    add   r0, r0, #0x10
    cmp   r4, #4
    addcc r0, r0, r4, lsl #10       ; + player * 0x4f0
    addcc r0, r0, r4, lsl #8
    subcc r0, r0, r4, lsl #4
    pop   {r4, pc}
mode_from_pad: .word mode_object - (b_pad_at1 + 8)
game_from_pad: .word game_object - (b_pad_at2 + 8)
pads_from_pad: .word pads - (b_pad_at3 + 8)

; ---- knock his bomb away, the way he faces: r6 the bytes, r7 the leader. Changes r0-r3, r12, s0-s2 ----
b_knock:
    push  {r4, lr}
    sub   sp, sp, #0x18
    mov   r0, r7
    bl    facing_of
    vldr  s0, [r0, #0x3c]           ; the way he faces: across
    vldr  s1, [r0, #0x5c]           ; and along
    vldr  s2, b_forward
    vmul  s0, s0, s2
    vmul  s1, s1, s2
    mov   r1, #0
    str   r7, [sp]                  ; "knocked away by him",
    vstr  s0, [sp, #4]              ; this fast a frame: across,
    ldr   r0, b_upward
    str   r0, [sp, #8]              ; up
    vstr  s1, [sp, #0xc]            ; and along;
    str   r1, [sp, #0x10]
    str   r1, [sp, #0x14]           ; and left in the state it is in
    ldrb  r2, [r6, #MODE]
    cmp   r2, #PUTTING
    streq r1, [sp, #4]              ; set down, not thrown: let go with no speed at all
    streq r1, [sp, #8]
    streq r1, [sp, #0xc]
    ldr   r0, [r6, #BOMB]
    mov   r1, sp
    ldr   r12, [r0]
    ldr   r12, [r12, #0xf4]
    blx   r12                       ; the game's own way of knocking a thing away
    add   sp, sp, #0x18
    pop   {r4, pc}
b_forward: .word BOMB_FORWARD
b_upward:  .word BOMB_UPWARD
