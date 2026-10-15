; His bombs: taking one out, lighting it, making it, and a pool of them on every map.
;
; A bomb of his is the game's second kind of bomb, a bomb rock underneath with a model of its own
; (bomb.py), which only Bingo Battle uses, so that bomb rocks are untouched. It is made through the
; kind's own manager, hung on him as a Pikmin carrying a bomb rock is hung, and lit as fire lights a
; bomb rock: a byte of the bomb, and how long its fuse is. A bomb that is let go restarts its waiting
; on the first frame it finds itself free, which clears that byte, so it is lit two frames later, and
; knocked away once more on the frame between.

; ==== the walk state, once a frame, where it moves him: r4 the state, r8 the leader, r11 his pad ====
.org ROOM_BOMBS
b_walk:                             ; stands for the call to the game's steering; r0-r2 are that call's
    push  {r5, r6, r7, lr}
    bl    steer
    ldr   r12, [r8, #0x81c]
    cmp   r12, #LINK                ; only the leader whose model is his
    popne {r5, r6, r7, pc}
    mov   r7, r8                    ; r7: the leader, as in the punch state
    ldr   r6, bytes_from_walk
b_walk_at:
    add   r6, pc, r6                ; r6 -> the bytes
    bl    b_light                   ; one thrown is knocked on, and lit when its frames are up
    ldrb  r0, [r6, #MODE]
    cmp   r0, #0
    blne  b_drop                    ; he left the punch state with a bomb out (knocked out of it, say): it goes, as thrown
    mov   r0, #0
    strb  r0, [r6, #ASKED]          ; (only this frame's press asks: the punch state reads this on entering)
    ldrb  r0, [r6, #PEND]
    cmp   r0, #0
    popne {r5, r6, r7, pc}          ; one is still waiting to be lit: no other yet
    ldr   r1, [r11, #0x238]         ; the controller behind the pad
    cmp   r1, #0
    ldrbne r1, [r1, #0xa]           ; what was pressed on it this frame
    tst   r1, #4                    ; down on the D-pad, which the game's table of buttons maps to nothing
    popeq {r5, r6, r7, pc}
    mov   r0, #1
    strb  r0, [r6, #ASKED]          ; a bomb is asked for
b_go_punch:                         ; (his bow's code comes in here too, with the same four registers kept)
    sub   sp, sp, #8                ; into the punch state, as the walk state goes into it itself
    mov   r0, sp
    mov   r1, r8
    bl    hold
    ldr   r5, [r4, #0x3c]           ; the leader's state machine
    ldr   r0, [r5, #0x44]
    cmp   r0, #0
    movne r0, r5
    blne  drop_pending
    mov   r0, r5
    mov   r1, #0x30
    bl    make_punch
    str   r0, [r5, #0x44]
    str   r5, [r0, #0x48]
    mov   r1, #0
    strh  r1, [r5, #0x58]
    mov   r0, sp
    bl    release
    add   sp, sp, #8
    pop   {r5, r6, r7, lr}
    b     walk_return               ; and the walk state is done for this frame, as after its own changes of state
bytes_from_walk: .word STATE - (b_walk_at + 8)

; ---- a bomb let go: a frame on it is knocked away once more (should its being let go have stopped it), and
; ---- when its frames are up it is lit. r6 the bytes, r7 the leader. Changes r0-r3, r12 and s0-s2 ----
b_light:
    ldrb  r0, [r6, #PEND]
    subs  r0, r0, #1
    bxmi  lr                        ; none is waiting
    strb  r0, [r6, #PEND]
    push  {r4, lr}
    mov   r4, r0
    bl    b_alive
    cmp   r0, #0
    popeq {r4, pc}                  ; it is gone
    cmp   r4, #0
    beq   b_lit
    cmp   r4, #BOMB_WAIT - 1
    bleq  b_knock
    pop   {r4, pc}
b_lit:
    ldr   r12, [r0]
    ldr   r12, [r12, #0x88]
    blx   r12                       ; its own state machine
    ldr   r1, b_fuse
    str   r1, [r0, #0xb4]           ; how long a fuse (seconds; below zero: until the clip of its lit state ends)
    mov   r1, #1
    strb  r1, [r0, #0xb2]           ; "light me": the byte fire sets on a bomb rock
    pop   {r4, pc}
b_fuse: .word BOMB_FUSE

; ---- the bomb he last took out, if it is still that bomb and has not gone off: r6 the bytes; r0 the bomb, or 0.
; ---- Changes r1 and r2 ----
b_alive:
    ldr   r0, [r6, #BOMB]
    cmp   r0, #0
    bxeq  lr                        ; he has taken none out on this map
    ldr   r1, [r6, #SERIAL]
    ldr   r2, [r0, #0x274]
    cmp   r1, r2
    movne r0, #0                    ; it is gone, and another has been made in its place
    bxne  lr
    ldrb  r2, [r0, #0x3c8]
    tst   r2, #3
    movne r0, #0                    ; it is finished already
    bx    lr

; ---- ... and is still hung on him: r6 the bytes, r7 the leader; r0 the bomb, or 0. Changes r1 and r2 ----
b_ours:
    push  {r4, lr}
    bl    b_alive
    cmp   r0, #0
    popeq {r4, pc}
    ldr   r1, [r0, #0x294]          ; what it hangs on
    add   r2, r7, #0x280            ; his own link
    cmp   r1, r2
    movne r0, #0
    pop   {r4, pc}

; ---- no bomb is out any more; one still on him goes, as thrown: r6 the bytes, r7 the leader ----
b_drop:
    push  {r4, lr}
    mov   r0, #0
    strb  r0, [r6, #MODE]
    bl    b_ours
    cmp   r0, #0
    blne  b_fly
    pop   {r4, pc}

; ---- let go of the bomb on him and send it flying, to be lit when its frames are up: r0 the bomb, r6 the
; ---- bytes, r7 the leader ----
b_fly:
    push  {r4, lr}
    add   r0, r0, #0x280            ; its link
    bl    unhang
    mov   r0, #BOMB_WAIT
    strb  r0, [r6, #PEND]
    bl    b_knock
    pop   {r4, pc}

; ---- make a bomb over his head and hang it on him: r6 the bytes, r7 the leader; r0 the bomb, or 0 if the map
; ---- has none to give. Changes r1-r3, r12, s0 and s1 ----
b_make:
    push  {r4, r5, lr}
    sub   sp, sp, #0x64
    ldr   r0, table_from_make
b_make_at1:
    add   r0, pc, r0
    str   r0, [sp, #8]              ; a name, held the way the game hands one over ...
    ldr   r0, name_from_make
b_make_at2:
    add   r0, pc, r0
    str   r0, [sp, #0xc]            ; ... the game's own name for the kind of bomb
    ldr   r0, managers_from_make
b_make_at3:
    ldr   r0, [pc, r0]              ; the game's managers, one for each kind of thing
    add   r1, sp, #8
    bl    find_manager              ; the one for that kind
    subs  r5, r0, #0
    beq   b_made                    ; the map has none
    mov   r0, r7
    ldr   r12, [r7]
    ldr   r12, [r12, #0x15c]
    blx   r12                       ; where he is
    ldr   r1, [r0]
    ldr   r2, [r0, #8]
    vldr  s0, [r0, #4]
    vldr  s1, b_above
    vadd  s0, s0, s1                ; that far above his feet
    ldr   r0, record_from_make
b_make_at4:
    add   r0, pc, r0
    str   r0, [sp, #0x10]           ; what a new thing is made from, as the game fills one in:
    mov   r4, #0
    str   r4, [sp, #0x14]
    str   r1, [sp, #0x18]           ; where,
    vstr  s0, [sp, #0x1c]
    str   r2, [sp, #0x20]
    str   r4, [sp, #0x24]           ; turned no way in particular,
    str   r4, [sp, #0x28]
    str   r4, [sp, #0x2c]           ; and placed by nothing in the map's files
    str   r4, [sp, #0x30]
    mov   r0, r5
    add   r1, sp, #0x10
    ldr   r12, [r5]
    ldr   r12, [r12, #0x30]
    blx   r12                       ; make one: the manager's own function for it
    subs  r5, r0, #0
    beq   b_made                    ; all the map's bombs of that kind are out
    add   r3, sp, #0x34             ; where it sits on him: three rows of four numbers,
    mov   r0, #0
    mov   r1, #0x2c
b_make_zero:
    str   r0, [r3, r1]
    subs  r1, r1, #4
    bpl   b_make_zero
    mov   r0, #0x3f800000
    str   r0, [r3]                  ; turned as he is,
    str   r0, [r3, #0x14]
    str   r0, [r3, #0x28]
    ldr   r0, b_above
    str   r0, [r3, #0x1c]           ; that far above his feet
    ldr   r0, b_ahead
    str   r0, [r3, #0x2c]           ; and that far ahead of him
    str   r4, [sp, #4]              ; following him, and no joint of his,
    add   r0, sp, #4
    str   r0, [sp]
    add   r0, r5, #0x280            ; the bomb's link
    mov   r1, #1                    ; carried
    mov   r2, r7                    ; by him
    bl    hang
    str   r5, [r6, #BOMB]
    ldr   r0, [r5, #0x274]
    str   r0, [r6, #SERIAL]         ; which of that bomb's lives this is
    mov   r0, r5
b_made:
    add   sp, sp, #0x64
    pop   {r4, r5, pc}
table_from_make:    .word string_table - (b_make_at1 + 8)
name_from_make:     .word mine_name - (b_make_at2 + 8)
managers_from_make: .word managers - (b_make_at3 + 8)
record_from_make:   .word birth_table - (b_make_at4 + 8)
b_above: .word BOMB_ABOVE
b_ahead: .word BOMB_AHEAD

; ==== a map's pools being set up, the last of the game's own just done: r6 the memory they are cut
; ==== from, r10 -> the game's managers, r11 the table of the game's holder of a string. The game sets
; ==== up a pool of its second kind of bomb in Bingo Battle alone; this does the same everywhere else ====
b_pools:                            ; stands for the call that closes the last pool's stretch of memory; r0 is that call's
    push  {r4, r5, r7, lr}
    bl    scope_end
    ldr   r0, bytes_from_pools
b_pools_at0:
    add   r0, pc, r0
    mov   r1, #0
    str   r1, [r0, #PEND]           ; a new map: no bomb is waiting, asked for or out (three bytes of one word),
    str   r1, [r0, #BOMB]           ; and the last map's is forgotten
    ldr   r0, game_from_pools
b_pools_at1:
    ldr   r0, [pc, r0]              ; the game
    ldr   r0, [r0, #0x2f0]
    cmp   r0, #3
    popeq {r4, r5, r7, pc}          ; Bingo Battle: the game has set this pool up itself
    sub   sp, sp, #0x18
    ldr   r4, name_from_pools
b_pools_at2:
    add   r4, pc, r4                ; the game's own name for the kind
    str   r11, [sp, #4]
    str   r4, [sp, #8]
    add   r0, sp, #0xc
    add   r1, sp, #4
    mov   r2, r6
    bl    scope_begin               ; a stretch of memory of its own, as each pool has
    ldr   r0, [r10]
    ldr   r7, [sp, #0x10]
    add   r1, sp, #4
    mov   r2, #1
    str   r11, [sp, #4]
    str   r4, [sp, #8]
    bl    manager_for               ; the kind's manager, made now if the map had none
    subs  r4, r0, #0
    beq   b_pools_done              ; (the game does not allow for there being no such kind; this does)
    mov   r3, #BOMB_POOL
    mov   r1, r7
    mov   r2, r7
    bl    make_pool                 ; so many of them, ready to be made
    ldrb  r0, [r4, #0x7e]
    cmp   r0, #0
    bne   b_pools_done
    ldr   r0, [r4]
    ldr   r3, [r0, #0x48]
    mov   r0, r4
    mov   r1, r7
    mov   r2, r7
    blx   r3                        ; and the kind's files loaded
    mov   r0, #1
    strb  r0, [r4, #0x7e]
b_pools_done:
    add   r0, sp, #0xc
    bl    scope_end
    add   sp, sp, #0x18
    pop   {r4, r5, r7, pc}
game_from_pools: .word game_object - (b_pools_at1 + 8)
name_from_pools: .word mine_name - (b_pools_at2 + 8)
bytes_from_pools: .word STATE - (b_pools_at0 + 8)
