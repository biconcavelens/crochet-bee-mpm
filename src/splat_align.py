"""Put the trained Gaussian splat into the SIMULATION frame (mm; y up; floor at FLOOR; bee centred at x = z = ZC; long axis
along z, face toward -z) and select the bee's Gaussians.  Runs locally (numpy only).

Cameras: the exact training cameras nerfstudio used (splat/train_cams.json, dumped by colab/dump_cams.py). Note: the exported
splat is NOT in the frame described by dataparser_transforms.json + transforms.json (it differs by a rotation and shift),
so those must not be used to place cameras.

Steps: look-at point of the orbit -> support plane (RANSAC, normal within 30 deg of the splat's vertical axis, oriented toward
the cameras) -> bee = Gaussians above the plane near the look-at point -> scale from the bee's height (47 mm) -> long axis
(PCA of the footprint) -> face = end with the dark (eye/mouth) Gaussians -> transform everything; crop far-away junk.
in : splat/splat.ply, splat/train_cams.json      out: splat/bee_sim.npz, splat/cams_sim.json, splat/presscam.json, splat/align_check.png
usage: python src/splat_align.py
"""
import json, numpy as np, cv2

FLOOR, ZC, H0 = 6.0, 64.0, 47.0                       # must match src/mpm_bee.py

def read_ply(p):
    with open(p, "rb") as f:
        names, n = [], 0
        while True:
            l = f.readline().decode().strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): names.append(l.split()[-1])
            elif l == "end_header": break
        a = np.frombuffer(f.read(n * 4 * len(names)), dtype=np.float32).reshape(n, len(names))
    return {k: a[:, i] for i, k in enumerate(names)}

def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)

def mat2q(M):
    w = np.sqrt(max(0, 1 + np.trace(M))) / 2
    return np.array([w, (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w), (M[1, 0] - M[0, 1]) / (4 * w)])

if __name__ == "__main__":
    v = read_ply("splat/splat.ply")
    means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    logs = np.stack([v[f"scale_{i}"] for i in range(3)], 1); quat = np.stack([v[f"rot_{i}"] for i in range(4)], 1)
    opac = v["opacity"]; op = 1 / (1 + np.exp(-opac)); dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
    nrest = len([k for k in v if k.startswith("f_rest_")])
    rest = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).reshape(len(dc), 3, -1).transpose(0, 2, 1)
    cams = json.load(open("splat/train_cams.json"))
    C = np.array([c["c2w"] for c in cams]); cam_pos = C[:, :, 3]; cam_dir = -C[:, :, 2]          # OpenGL: looks along -z

    A = np.zeros((3, 3)); b = np.zeros(3)
    for p, d in zip(cam_pos, cam_dir):
        P = np.eye(3) - np.outer(d, d); A += P; b += P @ p
    c = np.linalg.solve(A, b); dcam = np.median(np.linalg.norm(cam_pos - c, axis=1))
    print(f"{len(means)} Gaussians; look-at {c.round(3)}; median camera distance {dcam:.3f}")

    # support plane near the look-at point; normal within 30 deg of the splat's vertical axis (+z), facing the cameras
    up0 = np.array([0, 0, 1.0]) * np.sign(np.mean(cam_pos[:, 2] - c[2]) or 1)
    near = (np.linalg.norm(means - c, axis=1) < 0.6 * dcam) & (op > 0.5); P = means[near]
    rng = np.random.default_rng(0); best = (0, None); tol = 0.004 * dcam
    for _ in range(5000):
        i = rng.choice(len(P), 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
        if np.linalg.norm(n) < 1e-12: continue
        n /= np.linalg.norm(n); n *= np.sign(n @ up0)
        if n @ up0 < np.cos(np.radians(30)) or (c - P[i[0]]) @ n < 0: continue                  # bee must sit above it
        k = (np.abs((P - P[i[0]]) @ n) < tol).sum()
        if k > best[0]: best = (k, (n, P[i[0]]))
    n, p0 = best[1]; inl = np.abs((P - p0) @ n) < tol; p0 = P[inl].mean(0)
    n = np.linalg.svd(P[inl] - p0)[2][2]; n *= np.sign(n @ up0)
    print(f"plane: {inl.sum()} inliers; normal {n.round(3)} ({np.degrees(np.arccos(n @ up0)):.1f} deg from vertical)")

    h = (means - p0) @ n; cp = c - ((c - p0) @ n) * n
    rad = np.linalg.norm((means - cp) - np.outer((means - cp) @ n, n), axis=1)
    cand = (h > 0.004 * dcam) & (h < 0.4 * dcam) & (rad < 0.2 * dcam) & (op > 0.5)
    hb = np.percentile(h[cand], 98); scale = H0 / hb
    # every Gaussian of the bee moves with the MPM, whatever its opacity (a static faint copy would ghost); the platform and
    # the shadow on it (< 1.5 mm above the plane) stay put
    bee = (h > 1.5 / scale) & (h < hb * 1.12) & (rad < 0.2 * dcam)
    e1 = np.cross(n, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
    q = means[bee] - cp; xy = np.stack([q @ e1, q @ e2], 1); w, V = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
    long = V[0, 1] * e1 + V[1, 1] * e2; long /= np.linalg.norm(long)
    lum = (0.5 + 0.2820948 * dc).clip(0, 1) @ [0.3, 0.59, 0.11]
    along = (means[bee] - cp) @ long; dark = lum[bee] < np.percentile(lum[bee], 5)
    if np.median(along[dark]) > np.median(along): long = -long                                  # +long = back, face at -z
    side = np.cross(n, long)                                                                    # x = y cross z (right-handed)
    foot = np.median(xy, 0); cent = cp + foot[0] * e1 + foot[1] * e2
    R = np.stack([side, n, long])
    assert np.linalg.det(R) > 0
    rel = (means[bee] - cent) @ R.T * scale
    ext = np.percentile(rel, 99, 0) - np.percentile(rel, 1, 0)
    print(f"scale {scale:.1f} mm/unit; bee extent x {ext[0]:.1f}, y {ext[1]:.1f}, z {ext[2]:.1f} mm (hand-measured 45 x 47 x 70)")

    X = (means - cent) @ R.T * scale + np.array([ZC, FLOOR, ZC])
    keep = (np.abs(X[:, 0] - ZC) < 160) & (np.abs(X[:, 2] - ZC) < 160) & (X[:, 1] > FLOOR - 60) & (X[:, 1] < FLOOR + 140)   # crop junk
    qR = mat2q(R); Q = qmul(np.repeat(qR[None], len(quat), 0), quat / np.linalg.norm(quat, axis=1, keepdims=True))
    np.savez("splat/bee_sim.npz", means=X[keep].astype(np.float32), log_scales=(logs + np.log(scale))[keep].astype(np.float32),
             quats=Q[keep].astype(np.float32), opacity=opac[keep].astype(np.float32), sh_dc=dc[keep].astype(np.float32),
             sh_rest=rest[keep].astype(np.float32), bee=bee[keep], scale=scale)
    print(f"kept {keep.sum()} of {len(keep)} Gaussians inside the crop box; bee Gaussians: {bee[keep].sum()}")
    cs = [dict(file=cm["file"], R=(R @ np.array(cm["c2w"])[:, :3]).tolist(), t=((np.array(cm["c2w"])[:, 3] - cent) @ R.T * scale + [ZC, FLOOR, ZC]).tolist(),
               fx=cm["fx"], fy=cm["fy"], cx=cm["cx"], cy=cm["cy"], w=cm["w"], h=cm["h"]) for cm in cams]
    json.dump(cs, open("splat/cams_sim.json", "w"))
    # press-like camera in front of the face, 15 deg above horizontal (OpenGL c2w)
    th = np.radians(15); dist = 230.0; tgt = np.array([ZC, FLOOR + 22, ZC]); pos = tgt + dist * np.array([0, np.sin(th), -np.cos(th)])
    fwd = (tgt - pos) / dist; right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    json.dump(dict(R=np.stack([right, up, -fwd], 1).tolist(), t=pos.tolist(), fx=1400.0, cx=300.0, cy=400.0, W=600, H=800), open("splat/presscam.json", "w"))

    # check image: point-splat preview from the press camera (all / bee only) and from the side
    def pts_view(R_, t_, f_, cx_, cy_, W_, H_, mask):
        Xm = X[keep][mask]; pc = (Xm - t_) @ R_; z = -pc[:, 2]; ok = z > 1
        u = f_ * pc[ok, 0] / z[ok] + cx_; vv = -f_ * pc[ok, 1] / z[ok] + cy_
        img = np.full((H_, W_, 3), 255, np.uint8); col = (0.5 + 0.2820948 * dc[keep][mask][ok]).clip(0, 1)
        for i in np.argsort(-z[ok]):
            if 0 <= u[i] < W_ and 0 <= vv[i] < H_: cv2.circle(img, (int(u[i]), int(vv[i])), 2, tuple(int(x * 255) for x in col[i][::-1]), -1)
        return img
    pr = json.load(open("splat/presscam.json")); Rp = np.array(pr["R"]); tp = np.array(pr["t"])
    sideR = np.array([[0, 0, -1.0], [0, 1, 0], [1, 0, 0]]).T; sidet = np.array([ZC + 230, FLOOR + 22, ZC])     # from +x looking -x
    imgs = [pts_view(Rp, tp, 1400, 300, 400, 600, 800, np.ones(keep.sum(), bool)), pts_view(Rp, tp, 1400, 300, 400, 600, 800, bee[keep] & (op[keep] > 0.5)),
            pts_view(sideR, sidet, 1400, 300, 400, 600, 800, bee[keep] & (op[keep] > 0.5))]
    for im, lab in zip(imgs, ("scene, press camera", "bee only, press camera", "bee only, side")):
        cv2.putText(im, lab, (10, 30), 0, 0.8, (0, 0, 0), 2)
    cv2.imwrite("splat/align_check.png", np.hstack(imgs))
