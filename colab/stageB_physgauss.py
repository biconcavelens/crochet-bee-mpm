"""Stage B2 (Colab VM, ns env + taichi): PhysGaussian-style rendering of the fitted MPM press.
The bee's Gaussians are embedded in the MPM proxy: each is bound to its K nearest MPM particles in the settled rest state;
per frame its centre moves with their (inverse-distance weighted) displacement and its covariance is deformed by the local
deformation gradient F (least-squares fit of the neighbours' relative positions): Sigma' = F Sigma F^T  (Xie et al. 2024).
The platform/table Gaussians stay static; the pusher is a dark slab of Gaussians. Frames match render_compare.py timing.
in : /content/bee_sim.npz, /content/presscam.json, /content/code/{src,out}   out: /content/photoreal_sim.mp4
run: cd /content/code && /content/ns/bin/python /content/code/colab/stageB_physgauss.py
"""
import sys, json, os, subprocess, numpy as np, torch, gsplat, cv2
from scipy.spatial import cKDTree
sys.path.insert(0, "src")
import mpm_bee as M
from fit_mpm_dense import ts, es, ls, LAG, TC
import fit_dense as D

K_NN, SPEED, FPS_OUT = 8, 2.0, 30
dev = "cuda"
Z = np.load("/content/bee_sim.npz"); b = json.load(open("out/fit3d_dense_best.json"))
bee = Z["bee"]
def T(a): return torch.tensor(a, dtype=torch.float32, device=dev)
mean0 = T(Z["means"]); quat0 = T(Z["quats"]); scl0 = torch.exp(T(Z["log_scales"])); opa = torch.sigmoid(T(Z["opacity"]))
col = torch.cat([T(Z["sh_dc"])[:, None], T(Z["sh_rest"])], 1); deg = int(np.sqrt(col.shape[1])) - 1
bee_t = torch.tensor(bee, device=dev)

def quat2mat(q):
    q = q / q.norm(dim=-1, keepdim=True); w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
Rg = quat2mat(quat0); cov0 = Rg @ torch.diag_embed(scl0 ** 2) @ Rg.transpose(1, 2)

# camera matched to the real panel of render_compare.py: 600x520, 15 deg above horizontal, ~6.8 px/mm at the bee, the bee's
# lowest image point on the floor line (y = 477). Rendered 2x supersampled, then downsampled.
ZC0 = M.NG * M.DX / 2; SS = 2; W, H = 600, 520; FLOOR_Y, PXMM, DIST = 477, 320.0 / M.H0, 230.0
th = np.radians(15.0); tgt = np.array([ZC0, M.FLOOR + 20.0, ZC0]); pos = tgt + DIST * np.array([0, np.sin(th), -np.cos(th)])
fwd = (tgt - pos) / DIST; right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
w2c = np.eye(4); w2c[:3, :3] = np.stack([right, -up, fwd]); w2c[:3, 3] = -w2c[:3, :3] @ pos            # OpenCV camera
f_px = PXMM * DIST
pc = (Z["means"][bee] - pos) @ w2c[:3, :3].T; ylow = np.percentile(f_px * pc[:, 1] / pc[:, 2], 99.5)  # lowest bee point (cy = 0)
CY = FLOOR_Y - ylow
VIEW = T(w2c)[None]; KM = T([[f_px * SS, 0, W / 2 * SS], [0, f_px * SS, CY * SS], [0, 0, 1]])[None]

# pusher: a SOLID dark rounded slab (superellipsoid, ~96 x 72 x 24 mm, like the hand-held stone), built from thin surface
# discs (flat Gaussians tangent to the surface) with baked shading: ambient + diffuse + a glossy highlight toward the camera
# rounded box |x/a|^4 + |y/b|^4 + |z/c|^4 = 1. Samples: uniform (area-weighted) on the faces of the bounding box, slid onto
# the surface along the ray from the centre (F is homogeneous of degree 4, so p -> p * F(p)^(-1/4) lands exactly on it).
# Gap-free coverage (~2.7 discs/mm^2); normals = gradient of F. (A 1 mm grid shell left holes at the face centres.)
A_, B_, C_ = 48.0, 12.0, 36.0
rng_p = np.random.default_rng(0); faces = [(0, (B_, C_)), (1, (A_, C_)), (2, (A_, B_))]; hs = (A_, B_, C_)
ar = np.array([4 * u * v for _, (u, v) in faces]); qs = []
for (ax, (u, v)), a_ in zip(faces, ar):
    n_ = int(60000 * a_ / ar.sum() / 2); oth = [i for i in range(3) if i != ax]
    for sg in (-1, 1):
        q = np.zeros((n_, 3)); q[:, oth[0]] = rng_p.uniform(-u, u, n_); q[:, oth[1]] = rng_p.uniform(-v, v, n_); q[:, ax] = sg * hs[ax]; qs.append(q)
Q = np.vstack(qs); P = Q * (((Q[:, 0] / A_) ** 4 + (Q[:, 1] / B_) ** 4 + (Q[:, 2] / C_) ** 4) ** -0.25)[:, None]
N = np.stack([P[:, 0] ** 3 / A_ ** 4, P[:, 1] ** 3 / B_ ** 4, P[:, 2] ** 3 / C_ ** 4], 1); N /= np.linalg.norm(N, axis=1, keepdims=True)
L = np.array([-0.4, 0.8, -0.45]); L /= np.linalg.norm(L)
V = pos - (P + np.array([ZC0, B_, ZC0])); V /= np.linalg.norm(V, axis=1, keepdims=True)
Hh = (L + V) / np.linalg.norm(L + V, axis=1, keepdims=True)
shade = 0.30 + 0.70 * np.clip(N @ L, 0, 1); spec = np.clip((N * Hh).sum(1), 0, 1) ** 60
rgb = np.clip(np.array([0.16, 0.16, 0.17]) * shade[:, None] + 0.55 * spec[:, None], 0, 1)
qz = np.stack([1 + N[:, 2], -N[:, 1], N[:, 0], np.zeros(len(N))], 1)                                    # rotate +z onto the normal
qz[N[:, 2] < -0.999] = [0, 1, 0, 0]; qz /= np.linalg.norm(qz, axis=1, keepdims=True)
Rp = quat2mat(T(qz)); sc_p = torch.tensor([[1.1, 1.1, 0.12]], device=dev).repeat(len(P), 1)
pus_cov = Rp @ torch.diag_embed(sc_p ** 2) @ Rp.transpose(1, 2)
pus_m = T(P + np.array([ZC0, B_, ZC0])); pus_o = torch.full((len(P),), 0.99, device=dev)
pus_c = torch.zeros((len(P), col.shape[1], 3), device=dev); pus_c[:, 0] = (T(rgb) - 0.5) / 0.2820948

def render(means, covs, yp):
    m, c, o, cl = means, covs, opa, col
    if yp is not None:                                     # pusher on the bee (its bottom face at the pusher height yp)
        # parts outside the MPM proxy (metal ring, wing tips) follow the head up into the pusher's volume. In reality they
        # are squashed under it, hidden from this camera; hide them (flattening them leaves streaks, because 3DGS sorts
        # whole Gaussians by centre depth and wide flat ones pop in front of the pusher)
        inside = bee_t & (m[:, 1] > yp - 0.3)
        if inside.any(): o = o.clone(); o[inside] = 0.0
        pm = pus_m + T([0, yp, 0])
        m = torch.cat([m, pm]); c = torch.cat([c, pus_cov]); o = torch.cat([o, pus_o]); cl = torch.cat([cl, pus_c])
    img, _, _ = gsplat.rasterization(m, None, None, o, cl, VIEW, KM, W * SS, H * SS, sh_degree=deg, covars=c,
                                     backgrounds=T([[0.85, 0.87, 0.84]]))
    img = (img[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

state = {"ref": None, "prev": None, "k": 0}
TEND = float(sys.argv[1]) if len(sys.argv) > 1 else ts[-1]; NL = int(np.searchsorted(ts, TEND)) + 1   # optional short test run
vid = cv2.VideoWriter("/content/_raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (W, H))
crown = M.FLOOR + M.H0 + 1.0

def bind(xr):
    """bind every bee Gaussian to its K nearest MPM particles in the rest state (first snapshot = settled, unloaded)."""
    g = Z["means"][bee]; d, idx = cKDTree(xr).query(g, K_NN)
    w = 1.0 / np.maximum(d, 0.3); w /= w.sum(1, keepdims=True)
    state.update(idx=torch.tensor(idx, device=dev), w=T(w), xr=T(xr))
    nb = state["xr"][state["idx"]]; cen = (state["w"][..., None] * nb).sum(1, keepdim=True); dr = nb - cen
    Bm = torch.einsum("nk,nki,nkj->nij", state["w"], dr, dr) + 1e-3 * torch.eye(3, device=dev)
    state.update(dr=dr, Binv=torch.linalg.inv(Bm))
    print(f"bound {len(g)} bee Gaussians to the MPM proxy; median neighbour distance {np.median(d):.2f} mm", flush=True)

def deformed(x):
    xc = T(x); nb = xc[state["idx"]]; disp = (state["w"][..., None] * (nb - state["xr"][state["idx"]])).sum(1)
    cen = (state["w"][..., None] * nb).sum(1, keepdim=True); dc = nb - cen
    F = torch.einsum("nk,nki,nkj->nij", state["w"], dc, state["dr"]) @ state["Binv"]
    m = mean0.clone(); c = cov0.clone()
    m[bee_t] = mean0[bee_t] + disp; c[bee_t] = F @ cov0[bee_t] @ F.transpose(1, 2)
    return m, c

def on_snap(tm, x, cm, yp):
    if state["ref"] is None: state["ref"] = True; bind(x)
    snap = (tm, x.copy(), yp)
    if state["prev"] is not None:
        t0, x0, y0 = state["prev"]
        while state["k"] / (FPS_OUT / SPEED) < tm and state["k"] <= int(TEND * FPS_OUT / SPEED):
            to = state["k"] / (FPS_OUT / SPEED); a = (to - t0) / max(tm - t0, 1e-9)      # interpolate between snapshots
            xi = (1 - a) * x0 + a * x; yi = (1 - a) * y0 + a * yp
            m, c = deformed(xi); img = np.ascontiguousarray(render(m, c, yi if yi < crown else None))
            vid.write(img); state["k"] += 1
            if state["k"] % 300 == 0: print(f"  frame {state['k']}  t={to:.1f}s", flush=True)
    state["prev"] = snap

M.run(b["E0_Pa"], b["nu"], b["hh"], b["e0"], ts[:NL], es[:NL], ls[:NL], tc=TC, snap_n=4, snap_cb=on_snap, r_grad=b.get("r_grad", 1.0))
vid.release()
subprocess.run("ffmpeg -y -loglevel error -i /content/_raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 /content/photoreal_sim.mp4", shell=True, check=True)
print("PHYSGAUSS DONE", state["k"], "frames")
