"""Find press peaks from a height track and cut out the scale display around each one.
usage: python src/press_events.py <clipN>   (reads out/<clipN>_track.csv, writes events csv + crop sheet)
"""
import sys, csv
import cv2, numpy as np
from scipy.signal import find_peaks, medfilt

VIDEOS = {"clip1": "presses/VID_20261111_145007963.mp4", "clip2": "presses/VID_20261111_145024622.mp4",
          "clip3": "presses/VID_20261111_145044137.mp4", "clip4": "presses/VID_20261111_145102822.mp4",
          "clip5": "presses/VID_20261111_150749389.mp4"}
OFFS = (-0.7, -0.35, 0, 0.35, 0.7)  # seconds around the strain peak


def events(csv_path):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    t, h = d["t_s"], medfilt(d["bottom_px"] - d["top_px"], 5)
    rest = np.percentile(h, 90)
    strain = (rest - h) / rest
    pk, _ = find_peaks(strain, prominence=0.08, distance=int(0.8 / np.median(np.diff(t))))
    return t, strain, rest, pk


def display_crop(cap, tsec):
    cap.set(cv2.CAP_PROP_POS_MSEC, tsec * 1000)
    ok, f = cap.read()
    if not ok:
        return np.zeros((100, 200, 3), np.uint8)
    m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), (40, 120, 120), (85, 255, 255))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    x, y, w, h, _ = st[1 + np.argmax(st[1:, 4])]
    c = f[y:y + h, max(x, w // 3):x + w]  # right two thirds holds the digits
    return cv2.resize(c, (int(c.shape[1] * 90 / c.shape[0]), 90))


if __name__ == "__main__":
    name = sys.argv[1]
    t, strain, rest, pk = events(f"out/{name}_track.csv")
    cap = cv2.VideoCapture(VIDEOS[name])
    rows = []
    with open(f"out/{name}_events.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["press", "t_peak_s", "peak_strain", "rest_px"])
        for i, k in enumerate(pk, 1):
            w.writerow([i, round(t[k], 3), round(strain[k], 3), round(rest, 1)])
            tiles = []
            for o in OFFS:
                c = display_crop(cap, max(0, t[k] + o))
                cv2.putText(c, f"P{i} {o:+.2f}s", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                tiles.append(cv2.resize(c, (170, 90)))
            rows.append(np.hstack(tiles))
    cv2.imwrite(f"out/{name}_crops.png", np.vstack(rows))
    print(name, "presses:", len(pk), "peak strains:", np.round(strain[pk], 2))
