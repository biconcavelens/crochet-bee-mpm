"""Track the bee's eyes and mouth (black on yellow) in stabilized frames, as material markers for HOW the bee deforms.
usage: python src/track_face.py [step_frames]   -> out/face.csv (t_s, eyeL_x, eyeL_y, eyeR_x, eyeR_y, mouth_x, mouth_y, bot)
"""
import sys, cv2, numpy as np
sys.path.insert(0, "src")
from stabilize import VIDEO, REF_T, feats, homography, frame_at

ROI = (300, 820, 330, 760)          # x0, x1, y0, y1 (reference view): face area, below the pusher's lowest position

def find_face(w):
    x0, x1, y0, y1 = ROI
    hsv = cv2.cvtColor(w[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (12, 90, 60), (34, 255, 255))
    yel = cv2.morphologyEx(yel, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))        # fill the eye/mouth holes
    dark = ((hsv[..., 2] < 75) & (yel > 0)).astype(np.uint8)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
    eyes = [(cen[i][0], cen[i][1], st[i, 4]) for i in range(1, n)
            if 400 < st[i, 4] < 4500 and 0.8 < st[i, 3] / max(st[i, 2], 1) < 3.2]            # eye-like ovals (rounder when squashed)
    # mouth: only inside an eroded face region, so the dark shadow along the bee's bottom edge cannot merge with it
    inner = cv2.erode(yel, np.ones((25, 25), np.uint8))
    dm = ((hsv[..., 2] < 75) & (inner > 0)).astype(np.uint8); dm = cv2.morphologyEx(dm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    nm, labm, stm, cenm = cv2.connectedComponentsWithStats(dm)
    mouths = [(cenm[i][0], cenm[i][1], stm[i, 4]) for i in range(1, nm)
              if 200 < stm[i, 4] < 3500 and stm[i, 2] / max(stm[i, 3], 1) > 1.5]                 # wide, flat
    best = None
    for i in range(len(eyes)):
        for j in range(len(eyes)):
            a, b = eyes[i], eyes[j]
            if not (100 < b[0] - a[0] < 240 and abs(b[1] - a[1]) < 40 and 0.5 < a[2] / b[2] < 2.0): continue
            if best is None or a[2] + b[2] > best[0]: best = (a[2] + b[2], a, b)
    if best is None: return None
    _, a, b = best
    m = [q for q in mouths if a[0] < q[0] < b[0] and 30 < q[1] - (a[1] + b[1]) / 2 < 170]
    mx, my = (max(m, key=lambda q: q[2])[:2]) if m else (np.nan - x0, np.nan - y0)
    return a[0] + x0, a[1] + y0, b[0] + x0, b[1] + y0, mx + x0, my + y0


if __name__ == "__main__":
    step = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    fr = np.genfromtxt("out/frames.csv", delimiter=",", names=True)
    cap = cv2.VideoCapture(VIDEO); ref = feats(frame_at(cap, REF_T)); rows = []; miss = 0
    for k in range(0, len(fr), step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k); ok, f = cap.read()
        if not ok: break
        H, n = homography(ref, feats(f))
        if H is None or n < 60: miss += 1; continue
        r = find_face(cv2.warpPerspective(f, H, (1080, 1920)))
        if r is None: miss += 1; continue
        rows.append((fr["t_s"][k], *r, fr["bot_loose"][k]))
    np.savetxt("out/face.csv", rows, delimiter=",", header="t_s,eyeL_x,eyeL_y,eyeR_x,eyeR_y,mouth_x,mouth_y,bot", comments="", fmt="%.2f")
    print(f"face found in {len(rows)} frames, missed {miss}")
