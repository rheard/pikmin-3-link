; The twenty-two words of the game's own code that change. Nineteen become a call to the new code,
; which does what the word did for every other leader; three are a clip's number, in places only the
; leader with Alph's model is sent to. The other files have the far side of each.

; ==== the walk state ====
.org 0x60c4fe14                         ; the call to the steering, once a frame
    bl    w_walk
.org 0x60c520e8                         ; a press of A has found nothing to do: is there to be a punch at nothing?
    bl    s_air
.org 0x60c51420                         ; the same, where a press is taken the controls' other way
    bl    s_air

; ==== entering the punch state ====
.org 0x60c55978                         ; "the clip to play is the punch"
    bl    x_enter_a
.org 0x60c559d4                         ; the same in the state's second way in
    bl    x_enter_b

; ==== each frame of the punch state ====
.org 0x60c55a48                         ; "what did his animation pass this frame?", asked before the search
    bl    x_flags
.org 0x60c55aa4                         ; the radius of the sphere searched for an enemy, 10.0
    bl    x_reach
.org 0x60c55b14                         ; the furthest away a hit may land, 20.0
    bl    w_furthest
.org 0x60c55b64                         ; the search: what kind of thing it has found
    bl    x_type
.org 0x60c55c30                         ; the strength of the hit, 3.0
    bl    x_strength
.org 0x60c55f28                         ; "has the clip ended?"
    bl    x_update
.org 0x60c55fd0                         ; the same question on the path taken after a hit
    bl    x_update

; ==== entering the dodge, with the clip chosen ====
.org 0x60c56360
    bl    s_dodge

; ==== a map's pools being set up: the last of the game's own is done ====
.org 0x60a8c688
    bl    b_pools

; ==== a part of a leader or a Pikmin is handed its new clip's frame, as his speed changes ====
.org 0x60c8c3bc                         ; both parts at once: the upper body's
    bl    g_frame
.org 0x60c8c3cc                         ; ... and the legs'
    bl    g_frame
.org 0x60c8c43c                         ; one part
    bl    g_frame

; ==== someone left standing is given a clip to play once ====
.org 0x609bd0dc                         ; the call that starts it, with the clip just chosen
    bl    i_fidget
.org 0x609bcd74                         ; the one sitting is to lie down and sleep
    bl    i_sleep

; ==== someone left standing rests: the three places a sitting clip is picked, by which leader it is ====
.org 0x609bc9b8                         ; "Alph's sitting clip", as a rest begins
    mov   r0, #SIT_CLIP
.org 0x609bcca8                         ; the same, where a sleeper is sat up again
    mov   r0, #SIT_CLIP
.org 0x609bcda0                         ; the same, where one who stands is sat down
    mov   r0, #SIT_CLIP
