; Setting a bomb down, and the frame each part of him goes on from as his speed changes.
;
; As a leader speeds up or slows down, the game moves each part of him onto a new clip at the frame
; the old one was on. His stand is far longer than any clip of the leaders', so that frame is first
; brought round into the new clip's loop; otherwise his legs would stop on the last frame of a walk.

; ==== the frame a part's new clip starts on, where the game changes the clip as his speed changes:
; ==== r0 what plays his two parts, r1 which part, s0 the frame. Stands for the game's own function ====
.org ROOM_PUT
g_frame:
    ldrb  r2, [r0, #0x28]           ; are his two parts on clips of their own?
    add   r0, r0, #0x20
    cmp   r2, #0
    moveq r1, r2                    ; if not, the first plays both
    cmp   r1, #2
    addcc r0, r0, r1, lsl #2
    ldr   r0, [r0]                  ; what plays that part
    ldr   r1, [r0]
    ldr   r2, [r1, #0x38]           ; its "go to this frame" ...
    mov   r1, #1                    ; ... brought round into the clip's loop first, which the game's own call leaves out
    bx    r2

; ---- D-pad Down with a bomb held: he sets it down. r6 the bytes ----
b_put:
    mov   r0, #0
    strb  r0, [r6, #PLACE]          ; the first frame of it
    mov   r0, #PUTTING
    mov   r1, #PUT_CLIP
    b     b_play

; ---- each frame of setting it down: r6 the bytes, r7 the leader ----
b_putting:
    mov   r0, r7
    ldr   r12, [r7]
    ldr   r12, [r12, #0x19c]
    blx   r12                       ; stop: the call the walk state makes with the stick let go
    bl    b_ours
    cmp   r0, #0
    beq   b_steered                 ; it has left his hands (or is gone)
    ldrb  r1, [r6, #PLACE]          ; frames into setting it down
    cmp   r1, #PUT_FRAMES
    bcs   b_steered                 ; (past the frame he lets go on: it stays where that frame put it)
    add   r2, r1, #1
    strb  r2, [r6, #PLACE]
    add   r3, pc, r1, lsl #3        ; two words a frame, straight after the branch below
    ldr   r2, [r3, #16]
    str   r2, [r0, #0x330]          ; how far above his feet it sits on him now: its link's own place for it,
    ldr   r2, [r3, #20]
    str   r2, [r0, #0x340]          ; and how far ahead of him
    b     b_steered
    .words PUT_PATH
