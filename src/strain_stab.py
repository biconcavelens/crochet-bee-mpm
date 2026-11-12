"""Bee height (px) in stabilized frames at each force-read time -> out/steps_pairs.csv.
Height = vertical extent of the yellow mask (central 60% of columns) inside the bee ROI of the reference view.
"""
import sys, cv2, numpy as np
sys.path.insert(0, "src")
from stabilize import VIDEO, REF_T, feats, homography, frame_at

ROI = (280, 820, 250, 900)   # x0,x1,y0,y1 in reference view

def bee_height(img):
    x0, x1, y0, y1 = ROI
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (16, 120, 70), (32, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((31, 31), np.uint8))   # drops thin leaks (wood strip, ring)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    m = lab == 1 + np.argmax(st[1:, 4])
    rw = m.sum(1); rows = np.where(rw >= 0.4 * rw.max())[0]      # rows where the blob is wide
    xs = np.where(m.any(0))[0]
    return rows.min() + y0, rows.max() + y0, xs.max() - xs.min()


if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO); ref = feats(frame_at(cap, REF_T))
    reads = np.genfromtxt("out/steps_force_reads.csv", delimiter=",", names=True, dtype=None, encoding=None)
    rows = []
    for t, g, c in reads:
        hs = []
        for dt in (-0.1, 0, 0.1):
            f = frame_at(cap, t + dt)
            if f is None: continue
            H, n = homography(ref, feats(f))
            if H is None or n < 100: continue
            r = bee_height(cv2.warpPerspective(f, H, (1080, 1920)))
            if r: hs.append(r)
        if hs:
            top, bot, w = np.median(hs, axis=0); rows.append((t, g, c, top, bot, bot - top, w)); print(f"{t:5.0f} {g:5.0f} {c:6s} top={top:5.0f} bot={bot:5.0f} h={bot-top:5.0f} w={w:4.0f}")
    with open("out/steps_pairs.csv", "w") as fh:
        fh.write("t_s,grams,conf,top_px,bot_px,h_px,w_px\n")
        for r in rows: fh.write("%g,%g,%s,%.1f,%.1f,%.1f,%.1f\n" % r)


def bee_edges_loose(img):
    """Pressed-frame measure: top of yellow mask (= pusher contact edge), bottom of mask, central 60% of columns."""
    x0, x1, y0, y1 = ROI
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    m = cv2.morphologyEx(cv2.inRange(hsv, (16, 120, 70), (32, 255, 255)), cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    m = lab == 1 + np.argmax(st[1:, 4]); ys, xs = np.where(m)
    a, b = np.percentile(xs, [20, 80]); c = (xs >= a) & (xs <= b)
    return ys[c].min() + y0, ys[c].max() + y0
