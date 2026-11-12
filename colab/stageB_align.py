"""Stage B1 (Colab VM, ns env): put the trained splat into the SIMULATION frame (mm; y up; floor at FLOOR; bee centred at
x = z = ZC; long axis along z, face toward -z) and select the bee's Gaussians.
in : /content/export/splat.ply, /content/outputs/.../dataparser_transforms.json, /content/bee/transforms.json
out: /content/bee_sim.npz  (all scene Gaussians in sim coords + bee mask), /content/align_*.png (checks)
run: /content/ns/bin/python /content/code/colab/stageB_align.py
"""
import json, glob, numpy as np, torch, gsplat, imageio.v2 as imageio

FLOOR, ZC, H0, W0, L0 = 6.0, 64.0, 47.0, 45.0, 70.0          # must match src/mpm_bee.py

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

v = read_ply("/content/export/splat.ply")
means = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
logs = np.stack([v[f"scale_{i}"] for i in range(3)], 1); quat = np.stack([v[f"rot_{i}"] for i in range(4)], 1)   # wxyz
opac = v["opacity"]; dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], 1)
nrest = len([k for k in v if k.startswith("f_rest_")]); rest = np.stack([v[f"f_rest_{i}"] for i in range(nrest)], 1).reshape(len(dc), 3, -1).transpose(0, 2, 1)
print(f"{len(means)} Gaussians, SH rest coeffs {rest.shape[1]}")

# cameras in the splat frame (ns-export keeps the dataparser frame: p' = scale * (T @ [p,1]))
dp = json.load(open(sorted(glob.glob("/content/outputs/bee/splatfacto/*/dataparser_transforms.json"))[-1]))
T = np.array(dp["transform"]); s_dp = dp["scale"]
TJ = json.load(open("/content/bee/transforms.json"))
c2w = []
for fr in TJ["frames"]:
    M = np.array(fr["transform_matrix"]); R = T[:, :3] @ M[:3, :3]; t = s_dp * (T[:, :3] @ M[:3, 3] + T[:, 3])
    c2w.append((R, t, fr))
cam_pos = np.array([t for _, t, _ in c2w]); cam_dir = np.array([-R[:, 2] for R, _, _ in c2w])     # OpenGL: camera looks along -z

# object centre: least-squares point closest to all optical axes (the orbit looks at the bee)
A = np.zeros((3, 3)); b = np.zeros(3)
for p, d in zip(cam_pos, cam_dir):
    P = np.eye(3) - np.outer(d, d); A += P; b += P @ p
c = np.linalg.solve(A, b); dcam = np.median(np.linalg.norm(cam_pos - c, axis=1))
print("look-at centre", c.round(3), "median camera distance", round(dcam, 3))

# support plane: RANSAC on opaque Gaussians near the centre; normal oriented toward the cameras
op = 1 / (1 + np.exp(-opac)); near = (np.linalg.norm(means - c, axis=1) < 0.6 * dcam) & (op > 0.5)
P = means[near]; rng = np.random.default_rng(0); best = (0, None)
for _ in range(3000):
    i = rng.choice(len(P), 3, replace=False); n = np.cross(P[i[1]] - P[i[0]], P[i[2]] - P[i[0]])
    if np.linalg.norm(n) < 1e-12: continue
    n /= np.linalg.norm(n); d = (P - P[i[0]]) @ n; k = (np.abs(d) < 0.004 * dcam).sum()
    if k > best[0] and abs(n @ (c - P[i[0]])) < 0.3 * dcam: best = (k, (n, P[i[0]]))
n, p0 = best[1]
if np.mean((cam_pos - p0) @ n) < 0: n = -n
inl = np.abs((P - p0) @ n) < 0.004 * dcam; p0 = P[inl].mean(0)                                 # refine
u, sv, vt = np.linalg.svd(P[inl] - p0); n2 = vt[2] * np.sign(vt[2] @ n); n = n2
print(f"plane: {inl.sum()} inliers, normal {n.round(3)}")

# the bee: above the plane, within a vertical cylinder around the centre
h = (means - p0) @ n; cp = c - ((c - p0) @ n) * n
rad = np.linalg.norm((means - cp) - np.outer((means - cp) @ n, n), axis=1)
cand = (h > 0.002 * dcam) & (h < 0.5 * dcam) & (rad < 0.25 * dcam) & (op > 0.3)
hb = np.percentile(h[cand], 99); scale = H0 / hb                                               # bee height -> 47 mm
bee = cand & (h < hb * 1.15)
# in-plane axes: long axis by PCA of the bee footprint
e1 = np.cross(n, [1, 0, 0]); e1 = e1 if np.linalg.norm(e1) > 0.1 else np.cross(n, [0, 1, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
q = means[bee] - cp; xy = np.stack([q @ e1, q @ e2], 1); w, V = np.linalg.eigh(np.cov((xy - xy.mean(0)).T))
long = V[:, 1][0] * e1 + V[:, 1][1] * e2; side = np.cross(n, long)
# face = the end of the long axis holding the dark (eye/mouth) Gaussians
lum = (0.5 + 0.282 * dc).clip(0, 1) @ [0.3, 0.59, 0.11]
along = (means[bee] - cp) @ long; dark = lum[bee] < np.percentile(lum[bee], 5)
if np.median(along[dark]) > 0: long, side = -long, -side          # make +long point to the BACK (sim +z), face at -z
foot = np.median(xy, 0); cent = cp + foot[0] * e1 + foot[1] * e2
R = np.stack([side, n, long])                                      # rows: sim x, y, z axes in splat coords
ext = np.ptp((means[bee] - cent) @ R.T, axis=0) * scale
print(f"scale {scale:.2f} mm/unit; bee extent x {ext[0]:.1f} y {ext[1]:.1f} z {ext[2]:.1f} mm (measured 45 x 47 x 70)")

# transform every Gaussian into the sim frame
X = (means - cent) @ R.T * scale + np.array([ZC, FLOOR, ZC])
def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], 1)
def mat2q(M):
    w = np.sqrt(max(0, 1 + M[0, 0] + M[1, 1] + M[2, 2])) / 2
    return np.array([w, (M[2, 1] - M[1, 2]) / (4 * w), (M[0, 2] - M[2, 0]) / (4 * w), (M[1, 0] - M[0, 1]) / (4 * w)])
qR = mat2q(R if np.linalg.det(R) > 0 else -R)
if np.linalg.det(R) < 0: print("WARNING: left-handed frame, flipping side axis"); R[0] *= -1; X = (means - cent) @ R.T * scale + np.array([ZC, FLOOR, ZC]); qR = mat2q(R)
Q = qmul(np.repeat(qR[None], len(quat), 0), quat / np.linalg.norm(quat, axis=1, keepdims=True))
cams = [dict(R=R @ Rc, t=(tc - cent) @ R.T * scale + np.array([ZC, FLOOR, ZC]), file=fr["file_path"], fl=(fr.get("fl_x", TJ.get("fl_x")), fr.get("fl_y", TJ.get("fl_y"))),
             c=(fr.get("cx", TJ.get("cx")), fr.get("cy", TJ.get("cy"))), wh=(fr.get("w", TJ.get("w")), fr.get("h", TJ.get("h")))) for Rc, tc, fr in c2w]
np.savez("/content/bee_sim.npz", means=X.astype(np.float32), log_scales=(logs + np.log(scale)).astype(np.float32), quats=Q.astype(np.float32),
         opacity=opac.astype(np.float32), sh_dc=dc.astype(np.float32), sh_rest=rest.astype(np.float32), bee=bee, scale=scale)
json.dump([dict(R=cm["R"].tolist(), t=cm["t"].tolist(), file=cm["file"], fl=cm["fl"], c=cm["c"], wh=cm["wh"]) for cm in cams], open("/content/cams_sim.json", "w"))

# ---- checks: render from a training camera (compare with its photo) and from a press-like viewpoint ------------------
dev = "cuda"
G = dict(means=torch.tensor(X, dtype=torch.float32, device=dev), quats=torch.tensor(Q, dtype=torch.float32, device=dev),
         scales=torch.exp(torch.tensor(logs + np.log(scale), dtype=torch.float32, device=dev)), opacities=torch.sigmoid(torch.tensor(opac, device=dev)),
         colors=torch.cat([torch.tensor(dc, device=dev)[:, None], torch.tensor(rest, device=dev)], 1))
deg = int(np.sqrt(G["colors"].shape[1])) - 1
def render(c2w_R, c2w_t, fx, fy, cx, cy, W, H, mask=None):
    w2c = np.eye(4); w2c[:3, :3] = c2w_R.T; w2c[:3, 3] = -c2w_R.T @ c2w_t
    w2c = np.diag([1, -1, -1, 1]) @ w2c                                  # OpenGL camera -> OpenCV for gsplat
    g = {k: (t if mask is None else t[mask]) for k, t in G.items()}
    img, _, _ = gsplat.rasterization(g["means"], g["quats"], g["scales"], g["opacities"], g["colors"],
                                     torch.tensor(w2c, dtype=torch.float32, device=dev)[None],
                                     torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=dev)[None], W, H, sh_degree=deg)
    return (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
cm = cams[len(cams) // 3]; W, H = int(cm["wh"][0]) // 2, int(cm["wh"][1]) // 2
im = render(cm["R"], cm["t"], cm["fl"][0] / 2, cm["fl"][1] / 2, cm["c"][0] / 2, cm["c"][1] / 2, W, H)
photo = imageio.imread("/content/bee/" + cm["file"]); photo = photo[::2, ::2][:H, :W]
imageio.imwrite("/content/align_traincam.png", np.hstack([photo, im]))
# press-like camera: in front of the face, 15 deg above horizontal, looking at the bee centre
th = np.radians(15); dist = 230.0; tgt = np.array([ZC, FLOOR + 22, ZC])
pos = tgt + dist * np.array([0, np.sin(th), -np.cos(th)]); fwd = (tgt - pos) / dist
right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
Rp = np.stack([right, up, -fwd], 1)                                       # OpenGL c2w
imageio.imwrite("/content/align_presscam.png", np.hstack([render(Rp, pos, 1400, 1400, 300, 400, 600, 800), render(Rp, pos, 1400, 1400, 300, 400, 600, 800, torch.tensor(bee, device=dev))]))
json.dump(dict(R=Rp.tolist(), t=pos.tolist(), fx=1400, cx=300, cy=400, W=600, H=800), open("/content/presscam.json", "w"))
print("bee Gaussians:", int(bee.sum()), "of", len(bee)); print("ALIGN DONE")
