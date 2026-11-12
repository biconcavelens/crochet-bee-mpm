"""Render the fitted 3-D MPM bee (front view) as an mp4: particles coloured by compaction, pusher, live force trace
against the per-frame display reads. Uses the dense fit (fit_mpm_dense.py).
usage: python src/render_sim.py   -> out/sim_only.mp4
"""
import sys, json, glob, os, subprocess, cv2, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M
from fit_mpm_dense import ts, es, ls, LAG, TC
import fit_dense as D

b = json.load(open("out/fit3d_dense_best.json"))
f, _, snaps = M.run(b["E0_Pa"], b["nu"], b["hh"], b["e0"], ts, es, ls, tc=TC, snap_n=1)
PX = 8; X0, X1, Y1 = 34, 94, 66; W, H = (X1 - X0) * PX, Y1 * PX; PH = 190; FMAX = 1750
tg = D.runs[D.runs[:, 2] > 0]; tgt = D.t[tg[:, 0].astype(int)] - LAG          # display values, shifted by the display lag

def trace(cur):
    p = np.full((PH, W, 3), 255, np.uint8)
    xs = lambda a: int(10 + (W - 20) * a / ts[-1]); ys = lambda v: int(PH - 20 - (PH - 40) * v / FMAX)
    for a, v in zip(tgt, tg[:, 2]): cv2.circle(p, (xs(a), ys(v)), 1, (0, 0, 0), -1)
    cv2.polylines(p, [np.array([(xs(a), ys(max(v, 0))) for a, v in zip(ts, f)], np.int32)], False, (30, 120, 230), 2, cv2.LINE_AA)
    cv2.line(p, (xs(cur), 0), (xs(cur), PH), (0, 0, 200), 1)
    cv2.putText(p, "orange: 3-D sim force   black: scale display (lag-corrected)", (10, 16), 0, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return p

vid = cv2.VideoWriter("out/_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H + PH))
for tm, pts, cm, yp in snaps:
    img = np.full((H, W, 3), 245, np.uint8)
    px = ((pts[:, 0] - X0) * PX).astype(int); py = (H - pts[:, 1] * PX).astype(int)
    c = np.clip(cm / 0.5, 0, 1)
    col = np.stack([40 * (1 - c) + 200 * c, 190 * (1 - c) + 60 * c, 255 * (1 - c) + 60 * c], 1)          # BGR: yellow -> blue = compacted
    for i in np.argsort(pts[:, 2]): cv2.circle(img, (px[i], py[i]), 5, tuple(int(v) for v in col[i]), -1)
    cv2.rectangle(img, (0, 0), (W, int(H - yp * PX)), (90, 90, 90), -1)
    cv2.rectangle(img, (0, int(H - M.FLOOR * PX)), (W, H), (200, 200, 200), -1)
    cv2.putText(img, f"t={tm:5.1f}s  sim {np.interp(tm, ts, f):5.0f} g  strain {np.interp(tm, ts, es):+.2f}", (10, 22), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    vid.write(np.vstack([img, trace(tm)]))
vid.release()
FFMPEG = (glob.glob(os.path.expanduser("~/AppData/Local/Programs/Python/*/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe")) + ["ffmpeg"])[0]   # bundled copy if present, else ffmpeg on PATH
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", "out/_raw.mp4", "-c:v", "libx264", "-pix_fmt", "yuv420p", "out/sim_only.mp4"], check=True)
os.remove("out/_raw.mp4"); print(len(snaps), "frames")
