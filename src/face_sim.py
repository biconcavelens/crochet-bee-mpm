"""Measure the simulated bee the way track_face.py measures the real one: follow the material points where the eyes are
(front surface, 44% of the height, +-11.3 mm) and report below-eyes strain and eye-spacing change vs global strain.
usage (module): stats = face_metrics(E, nu, hh, e0, r_grad)   |   python src/face_sim.py  (current best fit vs real)
"""
import sys, json, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M
from fit_mpm_dense import ts, es, ls, TC

BINS = ((0.05, 0.15), (0.15, 0.25), (0.25, 0.32), (0.32, 0.45))
EYE_H, EYE_DX = 0.44, 11.3          # measured on the real bee (face.csv): eye height / bee height, half eye spacing (mm)
THETA = np.radians(28.0)            # camera elevation, from the platform disk ellipse and its 27 mm grid (minor/major ~0.47-0.49)
# the real metrics are measured in the IMAGE: image-up coordinate v = y cos(THETA) - z_cam sin(THETA), where z_cam points
# toward the camera (the face side, z < 0 here). Points bulging toward the lens look lower. The sim is measured the same way.

def eye_particles():
    c = np.array([M.NG * M.DX / 2, M.FLOOR + M.BODY[1], M.NG * M.DX / 2]); p = M.P_INIT - c
    y = EYE_H * M.H0 - M.BODY[1]; out = []
    for sx in (-1, 1):
        x = sx * EYE_DX; z = -M.BODY[2] * np.sqrt(max(1 - (x / M.BODY[0]) ** 2 - (y / M.BODY[1]) ** 2, 0))   # front surface (z < 0)
        d = np.linalg.norm(p - np.array([x, y, z]), axis=1); idx = np.where(d < 3.0)[0]
        idx = idx[idx % 2 == 0] // 2                                    # snapshots keep every 2nd particle
        out.append(idx)
    return out

THETAS = (10, 15, 20, 25, 30)

def face_metrics(E, nu, hh, e0, r_grad=1.0, f_out=None, hard="lin", thetas=THETAS):
    """returns {theta_deg: array (bins, 3): below-eyes strain, eye-spacing change, lowest-image-point drop (mm)}"""
    L, R = eye_particles(); rec = []
    zc = M.NG * M.DX / 2
    def cb(tm, x, cm, yp):
        row = [tm, x[L, 0].mean(), x[R, 0].mean()]
        for th in np.radians(thetas):
            v = x[:, 1] * np.cos(th) - (zc - x[:, 2]) * np.sin(th)
            row += [v[L].mean(), v[R].mean(), v.min()]
        rec.append(row)
    res = M.run(E, nu, hh, e0, ts, es, ls, tc=TC, snap_n=1, snap_cb=cb, r_grad=r_grad, hard=hard)
    if f_out is not None: f_out.append(res[:2])
    r = np.array(rec); tm = r[:, 0]; sp = r[:, 2] - r[:, 1]
    e = np.interp(tm, ts, es); load = np.interp(tm, ts, ls) > 0.5; out = {}
    for k, th in enumerate(thetas):
        vl, vr, vmin = r[:, 3 + 3 * k], r[:, 4 + 3 * k], r[:, 5 + 3 * k]
        ye = (vl + vr) / 2 - vmin                                       # eye height above the bee's lowest image point
        stats = []
        for lo, hi in BINS:
            m = load & (e >= lo) & (e < hi)
            stats.append((np.median(1 - ye[m] / ye[0]), np.median(sp[m] / sp[0] - 1), np.median(vmin[0] - vmin[m])) if m.any() else (np.nan,) * 3)
        out[th] = np.array(stats)
    return out

def real_metrics(px_per_mm=327.0 / 45.0):
    d = np.load("out/face_real.npz"); out = []
    for lo, hi in BINS:
        m = (d["e"] >= lo) & (d["e"] < hi)
        out.append((np.median(d["low"][m]), np.median(d["spread"][m]), np.median(d["drop"][m]) / px_per_mm))
    return np.array(out)

if __name__ == "__main__":
    b = json.load(open("out/fit3d_dense_best.json")); real = real_metrics()
    res = face_metrics(b["E0_Pa"], b["nu"], b["hh"], b["e0"], r_grad=b.get("r_grad", 1.0))
    print("real:      below-eyes", np.round(real[:, 0], 3).tolist(), " spacing", np.round(real[:, 1], 3).tolist(), " bottom drop mm", np.round(real[:, 2], 2).tolist())
    for th, st in res.items():
        print(f"sim {th:2d} deg: below-eyes", np.round(st[:, 0], 3).tolist(), " spacing", np.round(st[:, 1], 3).tolist(), " bottom drop mm", np.round(st[:, 2], 2).tolist())
