"""Large contrast-stretched display crops for reading the scale by eye.
usage: python src/crops_big.py <clipN> <first_press> <last_press>
"""
import sys
import cv2, numpy as np
sys.path.insert(0, "src")
from press_events import VIDEOS, events

def tile(cap, tsec, label):
    cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000); ok, f = cap.read()
    m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), (40, 120, 120), (85, 255, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    x, y, w, h, _ = st[1 + np.argmax(st[1:, 4])]
    g = f[y:y + h, x:x + w, 1].astype(float)
    lo, hi = np.percentile(g, 2), np.percentile(g, 85)
    g = np.clip((g - lo) / (hi - lo), 0, 1) ** 2  # square: pushes pale ghosts toward white
    g = cv2.resize((g * 255).astype(np.uint8), (420, 170))
    g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    cv2.putText(g, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return g

if __name__ == "__main__":
    name, a, b = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    t, strain, rest, pk = events(f"out/{name}_track.csv")
    cap = cv2.VideoCapture(VIDEOS[name]); rows = []
    for i in range(a, b + 1):
        k = pk[i - 1]
        rows.append(np.hstack([tile(cap, max(0, t[k] + o), f"P{i} {o:+.2f}s") for o in (-0.35, 0, 0.35)]))
    cv2.imwrite(f"out/{name}_big_{a}_{b}.png", np.vstack(rows))
