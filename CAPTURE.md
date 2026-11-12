# Capture checklist (crochet toy + sponge)

Goal: video that shows how each object deforms under known-ish force, so we can fit stiffness, damping, and yield.
Film both objects with the same protocol. Do the crochet toy and the sponge as separate sessions.

## 0. Before touching anything (pristine state)

- [ ] Weigh each object on the scale. Write the mass (g) down. We need it for density.
- [ ] Measure each object with a ruler: height, width, depth (cm). Write them down.
- [ ] **Orbit video (for optional Gaussian splatting later):** the object sits still on a plain surface in even, soft light.
      Walk slowly around it twice, once at table height looking slightly down and once higher. 60-90 s, 4K or 1080p, no zoom, no motion blur.
      Do this BEFORE any pressing, because pressing changes the shape for good.

## 1. Setup for the press videos

- [ ] Phone on a tripod or a stack of books, landscape, **side view**, lens at about the object's mid-height, 40-60 cm away.
- [ ] 1080p at **60 fps** (30 fps is the fallback). Lock focus and exposure (long-press on the screen). No zoom changes. Do not touch the phone while filming.
- [ ] Plain matte background, bright even light, no flickering lamps.
- [ ] Scale on a flat table. Object centered on the platform. **The scale's display must be readable in the same frame**, with the object above it. If it is too small, tell me and we'll adjust.
- [ ] Ruler in frame, in the same plane as the object, for pixel-to-cm scale.
- [ ] Pusher: a flat, rigid object with a flat bottom (hardcover book, cutting board, lid). Stick 2-3 high-contrast dots on its side so we can track its height.
- [ ] Sponge: draw a grid of dots (e.g. 4x3, about 1 cm apart) with a marker on the face that points at the camera. Keep the sponge dry. The crochet toy's stitches are already trackable.
- [ ] Put a small dot on the scale body as a fixed reference, so we can detect any camera drift.
- [ ] Note the scale's unit (g) and press the tare button with the object on it. Turn off any auto-off and keep the display awake.

## 2. Press protocol (the same for each object)

Each press: start filming, wait 2 s, press down about 3 s, hold the peak about 1 s, lift about 3 s, then lift the pusher fully clear and **keep filming 30-60 s** while the object rests. Do not touch it during the rest.
Re-tare the scale before each press with the object resting on it.

Go from gentle to severe. Rest 1-2 minutes and reshape the object gently between presses.

| ID | Depth (% of height) | Speed | Role |
|----|---------------------|-------|------|
| L1, L2, L3 | about 10-20% | slow | fit |
| LF | about 10-20% | fast (about 0.5 s down) | fit (rate dependence) |
| M1 | about 30-35% | slow | fit |
| M2 | about 30-35% | slow | **held-out, do not use for fitting** |
| D1 | about 50% | slow | fit |
| D2 | about 50% | slow | **held-out, do not use for fitting** |

Mark the target depths on the ruler or a strip of paper. They do not need to be exact. Just log what you actually did.

- [ ] After the last deep press, rest 60 s, then measure the object's height and width again (permanent change).
- [ ] Optional: a second orbit video of the damaged object.
- [ ] Optional: one dead-weight run. Put a known weight (e.g. a labeled 500 g water bottle) on the pusher, hold 30 s, remove it, film 60 s.

## 3. Log sheet

Keep a text file or notes app entry. One line per video:

```
file_name, object, press_id, approx_depth_pct, speed(slow/fast), notes
```

Keep the original video files unedited. Name them with the press ID (e.g. toy_L1.mp4).
