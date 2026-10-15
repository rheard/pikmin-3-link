; A charge of his sword, and the spin it ends in: after his bow's code, in the mod's own program file.
;
; A held through the first frames of a swing takes him into his own game's charge; let go once it has
; charged, he spins, and the punch state's search, turned round him to where the blade is on each
; frame, strikes everything round him once. Each function here stands in front of one of the bow's,
; as that stands in front of the sword's, and passes on to it whatever is not the spin's.

; ==== entering the punch state: r0 the animation, r2 which part of him, r3 the game's function that plays a clip, r4 the state ====
x_enter_a:
    ldr   r12, [r4, #0x5c]          ; the leader
    ldr   r1, [r12, #0x81c]
    cmp   r1, #LINK
    bne   w_enter_a                 ; another leader: as it was
    ldr   r12, x_bytes_a
x_enter_at_a:
    add   r12, pc, r12
    mov   r1, #0
    strb  r1, [r12, #SPIN_MODE]     ; a new punch state: no charge and no spin
    b     w_enter_a
x_enter_b:
    ldr   r12, [r4, #0x28]          ; the same, from the twin's side of the state
    ldr   r1, [r12, #0x81c]
    cmp   r1, #LINK
    bne   w_enter_b
    ldr   r12, x_bytes_b
x_enter_at_b:
    add   r12, pc, r12
    mov   r1, #0
    strb  r1, [r12, #SPIN_MODE]
    b     w_enter_b
x_bytes_a: .word STATE - (x_enter_at_a + 8)
x_bytes_b: .word STATE - (x_enter_at_b + 8)

; ==== each frame of the punch state, "has the clip ended?": r0 the animation, r4 the state, r7 the leader ====
x_update:
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    bne   w_update                  ; another leader: as it was
    ldr   r12, x_bytes_u
x_update_at:
    add   r12, pc, r12              ; r12 -> the bytes
    ldrb  r1, [r12, #PATCHED]
    cmp   r1, #0
    beq   w_update                  ; (the patch not seen to be here yet: the bow's code does the game's own call)
    ldrb  r1, [r12, #BOW_MODE]
    cmp   r1, #0
    bne   w_update                  ; his bow is out: the bow's code
    ldrb  r1, [r12, #SPIN_MODE]
    cmp   r1, #0
    bne   x_charged                 ; charging, spinning or straightening up: a frame of his own
    ldrb  r1, [r12, #MODE]
    cmp   r1, #0
    bne   w_update                  ; a bomb out as the frame begins: not a swing (whether or not its code lets it go this frame)
    push  {r4, r5, r6, lr}          ; (as that code keeps them, so that its own way out serves)
    mov   r6, r12                   ; r6 -> the bytes, as there
    bl    s_update                  ; a frame of a swing, by the sword's code
    mov   r5, r0                    ; what it answers
    mov   r0, r7
    bl    b_pad                     ; the pad of whoever plays him
    cmp   r0, #0
    ldrne r0, [r0, #0x200]          ; what is held on it
    and   r0, r0, #1                ; A (nobody plays him: as let go)
    ldrb  r1, [r6, #FRAMES]         ; frames into this swing: none or one as it begins
    cmp   r1, #1
    ldrbhi r2, [r6, #SPIN_HELD]
    andhi r0, r0, r2                ; held on every frame of it since
    strb  r0, [r6, #SPIN_HELD]
    cmp   r0, #0
    beq   s_done                    ; not held: just a swing
    cmp   r1, #HANDOVER
    bne   s_done                    ; held, not yet long enough
    mov   r0, #CHARGING             ; held long enough: he goes into the charge
    mov   r1, #CHARGE_CLIP
x_play:                             ; r0: what he does now; r1: its clip, on his whole body
    strb  r0, [r6, #SPIN_MODE]
    mov   r0, #0
    strb  r0, [r6, #SPIN_FRAMES]
    strb  r0, [r6, #SPIN_COUNT]     ; nothing struck yet
    ldr   r0, [r7, #0x374]          ; his animation
    ldr   r12, [r0]
    mov   r2, #2                    ; on the whole body
    ldr   r12, [r12, #0xc4]
    blx   r12                       ; play it
x_stay:
    bic   r5, r5, #0x80000000       ; the state goes on
    b     s_done
x_charged:
    push  {r4, r5, r6, lr}
    mov   r6, r12
    bl    anim_flags                ; what his animation passed this frame (below zero: its clip has ended)
    mov   r5, r0
    mov   r0, r7
    ldr   r12, [r7]
    ldr   r12, [r12, #0x19c]
    blx   r12                       ; he stands where he is: the call the walk state makes with the stick let go
    ldrb  r0, [r6, #SPIN_FRAMES]
    cmp   r0, #255
    addcc r0, r0, #1
    strbcc r0, [r6, #SPIN_FRAMES]   ; one frame further into it
    ldrb  r1, [r6, #SPIN_MODE]
    cmp   r1, #CHARGING
    bne   x_ending                  ; spinning or straightening up: until its clip has ended
    mov   r0, r7
    bl    b_pad
    cmp   r0, #0
    ldrne r0, [r0, #0x200]
    tst   r0, #1
    bne   x_stay                    ; A held: he holds the stance, which goes round
    ldrb  r0, [r6, #SPIN_FRAMES]
    cmp   r0, #SPIN_READY
    movcc r0, #LOWERING             ; let go before the charge is enough: he straightens up
    movcc r1, #LOWER_CLIP
    movcs r0, #SPINNING             ; let go after: he spins
    movcs r1, #SPIN_CLIP
    b     x_play
x_ending:
    cmp   r5, #0
    movlt r0, #0
    strblt r0, [r6, #SPIN_MODE]     ; its clip has ended: so does the state, as after a swing
    b     s_done
x_bytes_u: .word STATE - (x_update_at + 8)

; ==== the punch state's "what did his animation pass this frame?", before it looks for something to strike: r0 the
; ==== animation, r7 the leader ====
x_flags:                            ; (r12 alone is changed before the bow's code is called: the game's call is made with the rest as it was)
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    bne   w_flags                   ; another leader: as it was
    ldr   r12, x_bytes_f
x_flags_at:
    add   r12, pc, r12
    ldrb  r12, [r12, #SPIN_MODE]
    cmp   r12, #SPINNING
    bne   w_flags                   ; not his spin: as it was
    push  {r4, lr}
    bl    anim_flags
    ldr   r4, x_bytes_f2
x_flags_at2:
    add   r4, pc, r4
    ldrb  r1, [r4, #SPIN_FRAMES]    ; frames into the spin, counted at the end of each: its clip is on the frame after
    cmp   r1, #SPIN_UNTIL
    orrcc r0, r0, #0x400            ; a frame the blade can hit on: look for something to strike, as when a punch lands
    pop   {r4, pc}
x_bytes_f:  .word STATE - (x_flags_at + 8)
x_bytes_f2: .word STATE - (x_flags_at2 + 8)

; ==== the search's numbers: r7 the leader; s0, s2, s4 the way he faces (across, up, along). In his spin it hands back
; ==== r0 the radius, s18 how far out its middle is, and s0, s2, s4 the way to where the blade is (level) ====
x_reach:
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    bne   w_reach                   ; another leader: as it was
    ldr   r12, x_bytes_r
x_reach_at0:
    add   r12, pc, r12
    ldrb  r0, [r12, #SPIN_MODE]
    cmp   r0, #SPINNING
    bne   w_reach                   ; not his spin: as it was
    ldrb  r0, [r12, #SPIN_FRAMES]
    add   r0, r0, #1                ; the frame of the clip, and so where the blade is
    cmp   r0, #SPIN_LAST
    movhi r0, #SPIN_LAST
    ldr   r1, x_sweep_from
x_reach_at:
    add   r1, pc, r1                ; r1 -> the table: for each frame, how much of it is the way he faces and how much his left
    add   r1, r1, r0, lsl #3
    vldr  s1, [r1]
    vldr  s3, [r1, #4]
    vmul  s5, s0, s1                ; across: as he faces ...
    vmla  s5, s4, s3                ; ... and to his left, which runs along
    vmul  s6, s4, s1                ; along: as he faces ...
    vmls  s6, s0, s3                ; ... and to his left, which runs back across
    vmov  r1, s5
    vmov  s0, r1
    vmov  r1, s6
    vmov  s4, r1
    mov   r1, #0
    vmov  s2, r1                    ; level
    vldr  s18, x_out                ; the middle of the search this far out that way
    ldr   r0, x_radius
    bx    lr
x_sweep_from: .word x_sweep - (x_reach_at + 8)
x_bytes_r:    .word STATE - (x_reach_at0 + 8)
x_out:        .word SPIN_OUT
x_radius:     .word SPIN_RADIUS
x_sweep:
    .words SPIN_SWEEP

; ==== the strength of the blow: r7 the leader, r11 the thing found. Hands back r0; r1 is the game's to keep ====
x_strength:
    ldr   r12, [r7, #0x81c]
    cmp   r12, #LINK
    bne   w_strength                ; another leader: as it was
    ldr   r12, x_bytes_s
x_strength_at:
    add   r12, pc, r12
    ldrb  r0, [r12, #SPIN_MODE]
    cmp   r0, #SPINNING
    bne   w_strength                ; not his spin: as it was
    push  {r1}
    ldrb  r0, [r12, #SPIN_COUNT]
    cmp   r0, #SPIN_HITS
    addcc r1, r12, r0, lsl #2
    strcc r11, [r1, #SPIN_LIST]     ; kept: this spin does not strike it again
    addcc r0, r0, #1
    strbcc r0, [r12, #SPIN_COUNT]
    ldr   r0, x_power
    pop   {r1}
    bx    lr
x_bytes_s: .word STATE - (x_strength_at + 8)
x_power:   .word SPIN_POWER

; ==== the search, for each thing it finds, what kind of thing it is: r0 what the thing's own function said, r5 the thing,
; ==== r4 the state. Stands for the game's `and r0, r0, #0xff000000`; r1-r3 and r12 are the caller's to lose ====
x_type:
    and   r0, r0, #0xff000000       ; the game's own instruction
    ldr   r12, [r4, #0x5c]          ; the leader
    ldr   r12, [r12, #0x81c]
    cmp   r12, #LINK
    bxne  lr                        ; another leader: as it was
    ldr   r12, x_bytes
x_type_at:
    add   r12, pc, r12
    ldrb  r1, [r12, #SPIN_MODE]
    cmp   r1, #SPINNING
    bxne  lr                        ; not in his spin: as it was
    ldrb  r1, [r12, #SPIN_COUNT]
    add   r2, r12, #SPIN_LIST
x_type_next:
    subs  r1, r1, #1
    bxmi  lr                        ; not struck by this spin: as it was
    add   r3, r2, r1, lsl #2
    ldr   r3, [r3]
    cmp   r3, r5
    bne   x_type_next
    mov   r0, #0                    ; struck by it already: not taken for a thing to strike
    bx    lr
x_bytes: .word STATE - (x_type_at + 8)
