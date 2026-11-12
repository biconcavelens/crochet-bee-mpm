"""One pass over every frame of the stepped-hold clip: stabilize (homography to the reference view), measure the bee,
and keep a grayscale crop of the scale display for the digit reader.
usage: python src/stabilize_all.py
outputs: out/frames.csv (frame, t_s, inliers, top_loose, bot_loose, top_crown)   cache/lcd.npy (n, 110, 250) uint8
"""
import sys, cv2, numpy as np
sys.path.insert(0, "src")
from stabilize import VIDEO, REF_T, feats, homography, frame_at
from strain_stab import bee_height, bee_edges_loose

LCD = (0, 520, 1080, 1460)       # x0, x1, y0, y1 in the reference view (generous: the display face is not on the stabilized plane and drifts ~+-40 px)

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ref = feats(frame_at(cap, REF_T)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    lcd = np.zeros((n, (LCD[3] - LCD[2]) // 2, (LCD[1] - LCD[0]) // 2), np.uint8); rows = []
    for k in range(n):
        ok, f = cap.read()
        if not ok: break
        H, inl = homography(ref, feats(f))
        if H is None or inl < 60:
            rows.append((k, k / fps, inl, np.nan, np.nan, np.nan)); continue
        w = cv2.warpPerspective(f, H, (1080, 1920))
        try: tl, bl = bee_edges_loose(w)
        except Exception: tl = bl = np.nan
        r = bee_height(w); tc = r[0] if r else np.nan
        rows.append((k, k / fps, inl, tl, bl, tc))
        c = cv2.cvtColor(w[LCD[2]:LCD[3], LCD[0]:LCD[1]], cv2.COLOR_BGR2GRAY); lcd[k] = cv2.resize(c, (lcd.shape[2], lcd.shape[1]), interpolation=cv2.INTER_AREA)
        if k % 1000 == 0: print(k, "/", n, flush=True)
    np.save("cache/lcd.npy", lcd[:len(rows)])
    np.savetxt("out/frames.csv", rows, delimiter=",", header="frame,t_s,inliers,top_loose,bot_loose,top_crown", comments="", fmt="%.4f")
    print("done", len(rows))
