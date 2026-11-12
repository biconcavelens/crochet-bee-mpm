"""Measure the toy's visible height/width per frame from a side-view clip.

Toy = yellow (+ black stripe) pixels. The ruler/hand sit on top of it, so the top of the
yellow mask is the contact surface. Scale (mm/px) comes from the toy's known width.
usage: python src/track_height.py <video> <out.csv> [step]
"""
import sys, csv
import cv2, numpy as np

TOY_W_MM, TOY_H_MM = 45.0, 47.0  # measured by hand (front view: width x height)


def toy_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (18, 130, 130), (35, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))  # merge black stripes/eyes
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + np.argmax(st[1:, 4])
    return lab == i


def measure(video, step=2):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    rows, k = [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if k % step == 0:
            m = toy_mask(f)
            if m is not None:
                ys, xs = np.where(m)
                # central 60% of columns: robust to wings/ruler edge
                x0, x1 = np.percentile(xs, [20, 80])
                c = (xs >= x0) & (xs <= x1)
                rows.append((k / fps, ys[c].min(), ys[c].max(), xs.min(), xs.max(), int(m.sum())))
        k += 1
    return rows


if __name__ == "__main__":
    video, out = sys.argv[1], sys.argv[2]
    step = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    rows = measure(video, step)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "top_px", "bottom_px", "left_px", "right_px", "area_px"])
        w.writerows(rows)
    print(len(rows), "frames ->", out)
