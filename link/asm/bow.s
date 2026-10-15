; His bow, at the start of the mod's own program file.
;
; A click of the right stick puts him in the punch state with his bow out, by the bombs' way into it.
; He raises it and walks about aiming; A looses an arrow, which flies on ahead of him, and on each
; frame of its flight the punch state's own search looks for something to strike where its point is.
; A click again puts the bow away. Each function here stands in front of the sword's, and passes on
; to it whatever is not the bow's. The walk state's call first checks that the patch's own code is
; where this file expects it; if it is not, the game's own calls are made, and the rest of this file
; asks the byte that check leaves.

; ==== the walk state, once a frame, where it moves him: r0-r2 the game's own call's, r4 the state, r8 the leader, r11 his pad ====
.org PROGRAM_CODE
w_walk:
    ldr   r12, w_probe
w_walk_at0:
    ldr   r12, [pc, r12]            ; a word of the mod's own code, where its patch puts it
    ldr   r3, w_expect
    cmp   r12, r3
    ldr   r12, w_bytes0
w_walk_at1:
    add   r12, pc, r12              ; r12 -> the bytes
    moveq r3, #1
    movne r3, #0
    strb  r3, [r12, #PATCHED]       ; whether the patch is here
    bne   steer                     ; it is not: the game's own call
    ldr   r3, [r8, #0x81c]
    cmp   r3, #LINK
    bne   b_walk                    ; another leader: the bombs' code, which passes him on
    mov   r3, #0
    strb  r3, [r12, #BOW_MODE]      ; he is in the walk state: his bow is not out
    strb  r3, [r12, #BOW_ASKED]
    ldrb  r3, [r12, #BOW_KEPT]
    cmp   r3, #0
    bne   w_walk_keep               ; but he had it out when something ended the punch state (a blow): he takes it up again
    ldrb  r3, [r11, #9]             ; what was pressed this frame, the second byte of it
    tst   r3, #8                    ; a click of the right stick
    beq   b_walk
w_walk_keep:
    mov   r3, #1
    strb  r3, [r12, #BOW_ASKED]     ; his bow is asked for: the punch state reads this on entering
    push  {r5, r6, r7, lr}
    bl    steer                     ; (he is moved this frame as on any other)
    b     b_go_punch                ; and into the punch state, by the lines the bombs' code goes into it by

; ==== entering the punch state: r0 the animation, r2 which part of him, r3 the game's function that plays a clip, r4 the state ====
w_enter_a:
    ldr   r12, [r4, #0x5c]          ; the leader
    b     w_enter
w_enter_b:
    ldr   r12, [r4, #0x28]
w_enter:
    push  {r0, r2, r3, lr}
    ldr   r3, w_bytes1
w_enter_at:
    add   r3, pc, r3
    ldrb  r1, [r3, #PATCHED]
    cmp   r1, #0
    moveq r1, #0x42                 ; the patch is not here: the game's own instruction
    popeq {r0, r2, r3, pc}
    ldr   r1, [r12, #0x81c]
    cmp   r1, #LINK
    ldrbeq r1, [r3, #BOW_ASKED]
    cmpeq r1, #1
    beq   w_enter_bow
    pop   {r0, r2, r3, lr}
    b     s_enter                   ; a swing or a bomb: the sword's code (r12 the leader)
w_enter_bow:
    mov   r1, #0
    strb  r1, [r3, #BOW_ASKED]
    mov   r1, #BOW_RAISING
    strb  r1, [r3, #BOW_MODE]       ; he is raising his bow
    mov   r1, #1
    strb  r1, [r3, #BOW_KEPT]       ; and has it out until he puts it away
    mov   r2, #1
    bl    layer                     ; his legs go on following his speed
    pop   {r0, r2, r3, lr}
    mov   r1, #READY_CLIP
    mov   r2, #0                    ; and his upper body raises it
    bx    lr

; ==== each frame of the punch state, before it looks for something to hit: r0 the animation, r7 the leader.
; ==== Stands for the game's "what did his animation pass this frame?" ====
w_flags:
    push  {r4, lr}
    bl    anim_flags
    ldr   r3, w_bytes2
w_flags_at:
    add   r3, pc, r3
    ldr   r1, [r7, #0x81c]
    cmp   r1, #LINK
    ldrbeq r1, [r3, #BOW_MODE]
    cmpeq r1, #BOW_LOOSING
    popne {r4, pc}                  ; no arrow of his is in the air (the byte is only ever set with the patch here)
    ldrb  r1, [r3, #BOW_FLIGHT]
    cmp   r1, #250
    addcc r1, r1, #1
    strb  r1, [r3, #BOW_FLIGHT]     ; one frame further into its flight
    ldrb  r2, [r3, #BOW_HIT]
    cmp   r2, #0
    popne {r4, pc}                  ; it has struck something: it flies no further
    cmp   r1, #FLIGHT
    orrls r0, r0, #0x400            ; in flight: look for something for it to strike, as on the frame a punch lands
    pop   {r4, pc}

; ==== the search's numbers, each where the game set one: r7 the leader ====
w_reach:                            ; r0: the radius searched; s18: how far ahead of him; s20: how high its middle is
    ldr   r12, w_bytes3
w_reach_at:
    add   r12, pc, r12
    ldr   r0, [r7, #0x81c]
    cmp   r0, #LINK
    ldrbeq r0, [r12, #BOW_MODE]
    cmpeq r0, #BOW_LOOSING
    beq   w_reach_arrow
    ldrb  r0, [r12, #PATCHED]
    cmp   r0, #0
    bne   s_reach                   ; the sword's code: his sword's numbers, or the game's
    mov   r0, #0x41000000
    orr   r0, r0, #0x200000         ; the patch is not here: 10.0, the game's own
    bx    lr
w_reach_arrow:
    ldrb  r0, [r12, #BOW_FLIGHT]    ; which frame of its flight
    add   r1, pc, r0, lsl #2        ; (a word a frame, after the instructions below; the first frame is 1)
    ldr   r1, [r1, #16]
    vmov  s18, r1                   ; where its point is: that far ahead of him
    vldr  s1, w_rise
    vadd  s20, s20, s1              ; and the middle of what is searched that far above the level of his feet
    ldr   r0, w_radius
    bx    lr
    .words ARROW_REACHES
w_furthest:                         ; s16: the furthest a hit may land; r0 is the game's, kept
    push  {r0, r1}
    ldr   r12, w_bytes4
w_furthest_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #PATCHED]
    ldr   r0, [r7, #0x81c]
    cmp   r0, #LINK
    ldrbeq r0, [r12, #BOW_MODE]
    cmpeq r0, #BOW_LOOSING
    ldreq r12, w_far
    vmoveq s16, r12
    popeq {r0, r1}
    bxeq  lr
    cmp   r1, #0
    pop   {r0, r1}
    bne   s_furthest
    ldr   r12, w_theirs_far         ; the patch is not here: 20.0, the game's own
    vmov  s16, r12
    bx    lr
w_strength:                         ; r0: the strength of the blow. Something has been found to strike
    push  {r1}
    ldr   r12, w_bytes5
w_strength_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #PATCHED]
    ldr   r0, [r7, #0x81c]
    cmp   r0, #LINK
    ldrbeq r0, [r12, #BOW_MODE]
    cmpeq r0, #BOW_LOOSING
    moveq r0, #1
    strbeq r0, [r12, #BOW_HIT]      ; his arrow has struck: it flies no further
    ldreq r0, w_power
    popeq {r1}
    bxeq  lr
    cmp   r1, #0
    pop   {r1}
    bne   s_strength
    mov   r0, #0x40000000
    orr   r0, r0, #0x400000         ; the patch is not here: 3.0, the game's own
    bx    lr

; ==== each frame of the punch state, "has the clip ended?": r0 the animation, r4 the state, r7 the leader ====
w_update:
    ldr   r12, w_bytes6
w_update_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #PATCHED]
    cmp   r1, #0
    beq   anim_flags                ; the patch is not here: the game's own call
    ldr   r1, [r7, #0x81c]
    cmp   r1, #LINK
    ldrbeq r1, [r12, #BOW_MODE]
    movne r1, #0
    cmp   r1, #0
    beq   s_update                  ; his bow is not out: the sword's and the bombs' code
    push  {r4, r5, r6, lr}          ; (as that code keeps them, so that its own way out serves)
    mov   r6, r12                   ; r6 -> the bytes, as there
    bl    anim_flags
    mov   r5, r0                    ; what his animation passed this frame
    bl    b_light                   ; (a bomb thrown just before is still lit when its frames are up)
    mov   r0, r4
    bl    lock_on                   ; the lock-on button, taken as the walk state takes it every frame (the punch state does not)
    ldrb  r0, [r6, #BOW_MODE]
    cmp   r0, #BOW_LOOSING
    bcs   w_still                   ; loosing, or taking the next arrow: he stands where he is
    mov   r0, r4
    mov   r1, r7
    mov   r2, #0
    bl    steer                     ; raising it or aiming: he goes where the stick says, as in the walk state
    b     w_moved
w_still:
    mov   r0, r7
    ldr   r12, [r7]
    ldr   r12, [r12, #0x19c]
    blx   r12                       ; stop: the call the walk state makes with the stick let go
w_moved:
    ldrb  r0, [r6, #BOW_MODE]
    cmp   r0, #BOW_AIMING
    beq   w_aiming
    bhi   w_loosed
    cmp   r5, #0                    ; raising it: when that clip has ended, he aims
    bge   w_stay
w_aim:
    mov   r0, #BOW_AIMING
    mov   r1, #AIM_CLIP
    b     w_play
w_aiming:
    mov   r0, r7
    bl    b_pad                     ; the pad of whoever plays him
    cmp   r0, #0
    beq   w_away                    ; nobody does: the bow goes away
    ldrb  r1, [r0, #9]
    tst   r1, #8                    ; a click of the right stick again: he puts it away
    bne   w_away
    ldrb  r1, [r0, #8]
    tst   r1, #1                    ; A: he looses the arrow
    beq   w_stay
    mov   r0, #0
    strb  r0, [r6, #BOW_FLIGHT]     ; the first frame of its flight is the next
    strb  r0, [r6, #BOW_HIT]
    mov   r0, #BOW_LOOSING
    mov   r1, #LOOSE_CLIP
    b     w_play
w_loosed:
    cmp   r0, #BOW_TAKING
    beq   w_taking
    ldrb  r0, [r6, #BOW_HIT]
    cmp   r0, #0
    bne   w_take                    ; it has struck something
    cmp   r5, #0
    bge   w_stay                    ; it is still flying
w_take:
    mov   r0, #BOW_TAKING
    mov   r1, #NEXT_CLIP
    b     w_play
w_taking:
    cmp   r5, #0
    blt   w_aim                     ; the next arrow is on the string: he aims again
    b     w_stay
w_away:
    mov   r0, #0
    strb  r0, [r6, #BOW_MODE]
    strb  r0, [r6, #BOW_KEPT]       ; put away (or nobody plays him): the walk state does not take it up again
    orr   r5, r5, #0x80000000       ; the state ends, as after a swing
    b     s_done
w_play:                             ; r0: what he does next; r1: its clip
    strb  r0, [r6, #BOW_MODE]
    ldr   r0, [r7, #0x374]          ; his animation
    ldr   r12, [r0]
    mov   r2, #0                    ; on his upper body
    ldr   r12, [r12, #0xc4]
    blx   r12
w_stay:
    bic   r5, r5, #0x80000000       ; the state goes on
    b     s_done
w_probe:      .word b_walk + 4 - (w_walk_at0 + 8)
w_expect:     .word EXPECT
w_bytes0:     .word STATE - (w_walk_at1 + 8)
w_bytes1:     .word STATE - (w_enter_at + 8)
w_bytes2:     .word STATE - (w_flags_at + 8)
w_bytes3:     .word STATE - (w_reach_at + 8)
w_bytes4:     .word STATE - (w_furthest_at + 8)
w_bytes5:     .word STATE - (w_strength_at + 8)
w_bytes6:     .word STATE - (w_update_at + 8)
w_rise:       .word ARROW_RISE
w_radius:     .word ARROW_RADIUS
w_far:        .word ARROW_FAR
w_theirs_far: .word FURTHEST_THEIRS
w_power:      .word ARROW_POWER
