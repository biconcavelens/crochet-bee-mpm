"""Side-by-side video: real (stabilized) footage vs the fitted 3-D MPM simulation, same physical scale, plus the scale
display, the simulated force (lag-matched) and a force trace with the held-out cycles marked.
usage: python src/render_compare.py [t0 t1]      (optional window in seconds, for a quick test)
output: out/compare.mp4  (2x real time, 30 fps)
"""
import sys, json, glob, os, subprocess, cv2, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M
import fit_dense as D
from fit_mpm_dense import ts, es, ls, LAG, TC
from stabilize import VIDEO, REF_T, feats, homography, frame_at

PHOTO = None                                        # --photoreal <video>: right panel = PhysGaussian render (colab/stageB_physgauss.py)
if "--photoreal" in sys.argv:
    i = sys.argv.index("--photoreal"); PHOTO = sys.argv[i + 1]; del sys.argv[i:i + 2]
T0, T1 = (float(sys.argv[1]), float(sys.argv[2])) if len(sys.argv) > 2 else (0.0, ts[-1])
SPEED, FPS_OUT = 2.0, 30
PW, PH_TOP, MID, TR = 600, 520, 200, 190          # panel width, top-row height, middle-row height, trace height
FLOOR_Y = 477                                      # floor line (bee bottom) in both top panels
b = json.load(open("out/fit3d_dense_best.json"))

# ---- geometry of the real crop: centre on the bee in the reference view; scale = rest height px / 47 mm ----------------
cap = cv2.VideoCapture(VIDEO); fps = cap.get(cv2.CAP_PROP_FPS); ref_img = frame_at(cap, REF_T); ref = feats(ref_img)
hsv = cv2.cvtColor(ref_img[250:900, 280:820], cv2.COLOR_BGR2HSV)
ys_, xs_ = np.where(cv2.inRange(hsv, (16, 120, 70), (32, 255, 255)) > 0)
cx_px, bot_px = int(np.median(xs_)) + 280, int(np.percentile(ys_, 99.5)) + 250
S = 320.0 / M.H0                                   # px per mm: rest height 320 px / 47 mm (also gives the right width, ~306 px)
cx0, cy0 = cx_px - PW // 2, bot_px - FLOOR_Y
print(f"real crop x {cx0}..{cx0+PW}, y {cy0}..{cy0+PH_TOP}; {S:.2f} px/mm")

# ---- simulated force (first pass, no snapshots) for the trace and the numbers -----------------------------------------
f_sim, _ = M.run(b["E0_Pa"], b["nu"], b["hh"], b["e0"], ts, es, ls, tc=TC)
tg = D.runs[D.runs[:, 2] > 0]; t_disp = D.t[tg[:, 0].astype(int)] - LAG
hold_spans = [(D.t[D.runs[D.cyc_of_run == c][:, 0].astype(int)].min(), D.t[D.runs[D.cyc_of_run == c][:, 1].astype(int)].max()) for c in sorted(D.HOLD)]
cycle_at = lambda tm: int(D.cyc_of_run[max(np.searchsorted(D.t[D.runs[:, 0].astype(int)], tm) - 1, 0)])

def trace(cur):
    p = np.full((TR, 2 * PW, 3), 255, np.uint8); W = 2 * PW
    xs = lambda a: int(40 + (W - 60) * a / ts[-1]); ys = lambda v: int(TR - 22 - (TR - 45) * v / 1750)
    for a0, a1 in hold_spans: cv2.rectangle(p, (xs(a0), 18), (xs(a1), TR - 20), (225, 225, 255), -1)
    for v in (0, 500, 1000, 1500): cv2.putText(p, str(v), (2, ys(v) + 4), 0, 0.35, (120, 120, 120), 1, cv2.LINE_AA)
    for a, v in zip(t_disp, tg[:, 2]): cv2.circle(p, (xs(a), ys(v)), 1, (0, 0, 0), -1)
    cv2.polylines(p, [np.array([(xs(a), ys(max(v, 0))) for a, v in zip(ts, f_sim)], np.int32)], False, (30, 120, 230), 2, cv2.LINE_AA)
    cv2.line(p, (xs(cur), 14), (xs(cur), TR - 18), (0, 0, 200), 1)
    cv2.putText(p, "force (g):  black = scale display (lag-corrected)   orange = 3-D simulation   pink = held-out cycles (not used for fitting)",
                (40, 13), 0, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(p, "time (s) ->", (W - 90, TR - 4), 0, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    return p

# the sim is drawn through the REAL camera: elevation THETA (fitted from how far the bee's lowest image point drops under
# load, face_sim.py), so both panels show the same view. image-up v = y cos - z_cam sin (z_cam: toward the camera).
THETA = np.radians(15.0); ZC = M.NG * M.DX / 2      # camera elevation: estimates range 5-28 deg (bee proportions, disk fit,
                                                     # bottom-edge drop, grid); 15 deg is the middle. Not calibrated.
def v_img(p): return p[:, 1] * np.cos(THETA) - (ZC - p[:, 2]) * np.sin(THETA)
V0 = v_img(M.P_INIT).min()                         # lowest image point of the resting bee -> the floor line
LIGHT = np.array([-0.35, 0.75, 0.55]); LIGHT /= np.linalg.norm(LIGHT)
PUSHER_FRONT = 35.0                                # mm, how far the pusher reaches toward the camera (it hides the head top)

def face_paint():
    """black 'eyes' and 'mouth': the particles that start where the real ones are (face.csv: eyes at 44% height,
    24 mm apart, ~5x9 mm ovals; mouth at 11% height, ~13 mm wide), on the front surface. Indices into the
    every-2nd-particle snapshots. They are material points, so they deform with the bee."""
    c = np.array([ZC, M.FLOOR + M.BODY[1], ZC]); p = M.P_INIT - c; a = M.BODY
    zs = -a[2] * np.sqrt(np.clip(1 - (p[:, 0] / a[0]) ** 2 - (p[:, 1] / a[1]) ** 2, 0, 1))      # front surface depth here
    front = (p[:, 2] < 0) & (p[:, 2] - zs < 2.5) & (np.arange(M.N) < M.N_BODY)
    # the real fractions (eyes 44%, mouth 11% of the bee's height above its lowest point) are IMAGE fractions: find the
    # 3-D height on the front surface whose projection lands at that fraction of the projected bee
    vb = v_img(M.P_INIT[:M.N_BODY]); vlo, vhi = vb.min(), vb.max()
    def y_for(frac, x):
        ys = np.linspace(-a[1] * 0.98, a[1] * 0.98, 400)
        z = -a[2] * np.sqrt(np.clip(1 - (x / a[0]) ** 2 - (ys / a[1]) ** 2, 0, 1))
        v = (ys + c[1]) * np.cos(THETA) - (-z) * np.sin(THETA)
        return ys[np.argmin(abs((v - vlo) / (vhi - vlo) - frac))]
    ye, ym = y_for(0.44, 12.0), y_for(0.11, 0.0)
    eyes = (((np.abs(p[:, 0]) - 12.0) / 2.5) ** 2 + ((p[:, 1] - ye) / 4.5) ** 2 < 1)
    mouth = (np.abs(p[:, 0]) < 6.5) & (np.abs(p[:, 1] - (ym + 0.03 * p[:, 0] ** 2)) < 1.2)
    return (front & (eyes | mouth))[::2]
FACE = face_paint()
WING = (np.arange(M.N) >= M.N_BODY)[::2]

def sim_panel(pts, cm, yp):
    img = np.full((PH_TOP, PW, 3), (205, 212, 200), np.uint8)
    v = v_img(pts); px = (PW / 2 + (pts[:, 0] - ZC) * S).astype(int); py = (FLOOR_Y - (v - V0) * S).astype(int)
    # simple shading so it reads as a solid: normal ~ direction from the body centre, scaled by the current extents
    q = np.stack([pts[:, 0] - ZC, pts[:, 1], ZC - pts[:, 2]], 1); cen = np.median(q, 0); ext = np.percentile(np.abs(q - cen), 98, axis=0)
    n = (q - cen) / ext ** 2; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    b = 0.45 + 0.55 * np.clip(n @ LIGHT, 0, 1)
    c = 0.5 * np.clip(cm / 0.5, 0, 1)              # compaction tint (subtle)
    col = (np.array([40, 190, 255]) * b[:, None]) * (1 - c[:, None]) + np.array([200, 70, 60]) * c[:, None]
    col[FACE] = (20, 20, 20)
    col[WING] = col[WING] * 0 + np.array([200, 225, 232]) * b[WING, None]        # cream wings, as on the real bee
    depth = pts[:, 1] * np.sin(THETA) + (ZC - pts[:, 2]) * np.cos(THETA)          # toward the camera
    for i in np.argsort(depth): cv2.circle(img, (px[i], py[i]), 4, tuple(int(u) for u in col[i]), -1)
    if yp < M.FLOOR + M.H0 + 1.0:                  # draw the pusher only while it is on the bee (the hand takes it away)
        edge = yp * np.cos(THETA) - PUSHER_FRONT * np.sin(THETA)                   # pusher's lower front edge in the image
        cv2.rectangle(img, (0, 0), (PW, int(FLOOR_Y - (edge - V0) * S)), (45, 45, 45), -1)
    cv2.putText(img, "yellow = crochet, blue tint = compacted stuffing", (10, PH_TOP - 12), 0, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return img

state = {"prev": None, "k": 0, "cur_frame": -1, "frame": None, "H": None}
k_last = int(T1 * FPS_OUT / SPEED)                 # output frame k shows real time k * SPEED / FPS_OUT
RAW = "out/_raw_photo.mp4" if PHOTO else "out/_raw.mp4"           # separate temp files: both modes can run at once
vid = cv2.VideoWriter(RAW, cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (2 * PW, PH_TOP + MID + TR))
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def real_at(tm):
    want = int(round(tm * fps))
    while state["cur_frame"] < want:
        cap.grab(); state["cur_frame"] += 1
    ok, fr = cap.retrieve()
    H, n = homography(ref, feats(fr))
    if H is not None and n >= 60: state["H"] = H
    return cv2.warpPerspective(fr, state["H"], (1080, 1920))

def emit(tm, snap):
    w = real_at(tm)
    real = w[cy0:cy0 + PH_TOP, cx0:cx0 + PW].copy()
    cv2.line(real, (0, FLOOR_Y), (PW, FLOOR_Y), (200, 200, 200), 1)
    sim = snap if PHOTO else sim_panel(*snap[1:])
    lab_sim = "PHOTOREAL SIM: MPM + GAUSSIAN SPLAT" if PHOTO else "3-D MPM SIMULATION (fitted, same camera angle)"
    for im, lab in ((real, "REAL (stabilized video)"), (sim, lab_sim)):
        cv2.rectangle(im, (0, 0), (PW, 30), (0, 0, 0), -1); cv2.putText(im, lab, (10, 21), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    disp = cv2.resize(w[1140:1380, 0:520], (int(520 * MID / 240), MID))
    mid = np.full((MID, 2 * PW, 3), 30, np.uint8); mid[:, :disp.shape[1]] = disp
    c = cycle_at(tm); held = c in D.HOLD
    x0 = disp.shape[1] + 30
    cv2.putText(mid, f"t = {tm:6.2f} s    (2x speed)", (x0, 40), 0, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(mid, f"simulated scale reading: {max(np.interp(tm - LAG, ts, f_sim), 0):5.0f} g", (x0, 90), 0, 0.8, (80, 170, 255), 2, cv2.LINE_AA)
    cv2.putText(mid, "(left: the real scale display)", (x0, 125), 0, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(mid, f"press cycle {c}" + ("   HELD OUT - prediction" if held else "   used for fitting"), (x0, 170), 0, 0.65,
                (120, 120, 255) if held else (200, 200, 200), 2 if held else 1, cv2.LINE_AA)
    vid.write(np.vstack([np.hstack([real, sim]), mid, trace(tm)]))

def on_snap(tm, x, cm, yp):
    snap = (tm, x.copy(), cm.copy(), yp)
    if state["prev"] is not None:
        while state["k"] / (FPS_OUT / SPEED) < tm and state["k"] <= k_last:
            to = state["k"] / (FPS_OUT / SPEED)
            if to >= T0: emit(to, state["prev"])
            state["k"] += 1
            if state["k"] % 300 == 0: print(f"  t={to:.1f}s", flush=True)
    state["prev"] = snap

if PHOTO:                                           # photoreal frames are rendered at exactly these output times
    pv = cv2.VideoCapture(PHOTO)
    while state["k"] <= k_last:
        ok, fr = pv.read()
        if not ok: break
        to = state["k"] / (FPS_OUT / SPEED)
        if to >= T0: emit(to, np.ascontiguousarray(cv2.resize(fr, (PW, PH_TOP))))
        state["k"] += 1
        if state["k"] % 300 == 0: print(f"  t={to:.1f}s", flush=True)
else:
    M.run(b["E0_Pa"], b["nu"], b["hh"], b["e0"], ts, es, ls, tc=TC, snap_n=4, snap_cb=on_snap)
vid.release()
FFMPEG = (glob.glob(os.path.expanduser("~/AppData/Local/Programs/Python/*/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe")) + ["ffmpeg"])[0]   # bundled copy if present, else ffmpeg on PATH
OUT = "out/compare_photoreal.mp4" if PHOTO else "out/compare.mp4"
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", RAW, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", OUT], check=True)
os.remove(RAW); print("done", state["k"], "output frames")
