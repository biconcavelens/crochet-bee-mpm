"""3-D MLS-MPM of the bee (ellipsoid body + two wing blobs) pressed by a flat rigid pusher.  Taichi, GPU.

Units: mm, g, s  (1 Pa = 1 g/(mm s^2); 1 gf = 9810 force units).
Material: fixed-corotated hyperelastic, nu fixed, E_eff = E0 * (1 + hh * cmax) where cmax = deepest compaction (1-J) this
particle has seen (stuffing-compaction hardening, the 3-D version of model D in fit_steps.py).  No plasticity, no damping
(only a small numerical grid damping, DAMP).  Pusher = plane driven by the measured strain trajectory.  Floor and pusher
contacts are Coulomb friction (coefficient mu_fric, same for both) with free separation (no sticking, no adhesion).
Gravity on (9810 mm/s^2), bee mass 10 g (weighed).  The bee first settles under its own weight; the scale was tared with the
bee on it, so the force "read by the scale" = vertical floor reaction minus the bee's weight, averaged over the last 25% of
each interval.  The pusher trajectory is measured from the SETTLED crown, as in the video.
Quasi-static: the real timeline is compressed by TC (rate-independent material, so only inertia is affected; Mach ~1e-2).
"""
import numpy as np, taichi as ti

ti.init(arch=ti.gpu, default_fp=ti.f32, random_seed=0)
NG, DX = 64, 2.0
INV = 1.0 / DX
FLOOR = 6.0
H0 = 47.0                       # mm, rest height of the bee
MASS = 10.0                     # g, weighed on the scale
GRAV = 9810.0                   # mm/s^2
BODY = np.array([22.5, 23.5, 35.0])          # semi-axes (x width, y height, z length), mm
# wings: centres in BODY-CENTRED coordinates (y = 0 at mid-height; crown at +23.5). Measured on the reference frame:
# lobes ~15 mm either side of the centre line, behind the face, reaching ~4.5 mm above the crown, lower part merged
# into the head. Semi-axes >= 4 mm (2 grid cells) and a ~5 mm deep overlap with the body, so the 2 mm MPM grid keeps
# them attached (a thinner neck gets pinched off numerically).
WINGS = [(np.array([15.0, 20.0, 8.0]), np.array([6.5, 8.0, 4.0])), (np.array([-15.0, 20.0, 8.0]), np.array([6.5, 8.0, 4.0]))]
GF = 9810.0                     # force units per gram-force
DAMP = 400.0                    # 1/s grid velocity damping while loaded (numerical, not a material property). Tied to the time
                                # compression TC (fit_mpm_dense.py): DAMP*TC = 8e4 keeps the rate-dependent drag the same; at TC=200
                                # (DAMP 100) the bee's ~6 ms elastic ring was stretched to a visible ~1 s 'jelly' wobble in holds.
DAMP_FREE = 1600.0              # 1/s while the pusher is off the bee: ~critical, because the real bee shows no rebound ringing.
                                # (applied only to unloaded intervals, so it adds no drag to the loaded forces used for fitting)
VMAX = 300.0                    # mm/s, max pusher speed in sim time


def make_particles(density=0.8, seed=0):
    """uniform particles (per mm^3) inside body + wing ellipsoids (wing parts that overlap the body are dropped)."""
    rng = np.random.default_rng(seed)
    def sample(c, a):
        n = int(density * 4 / 3 * np.pi * a.prod() * 1.9)
        p = rng.uniform(-1, 1, (n, 3)) * a
        return (p[((p / a) ** 2).sum(1) <= 1] + c)
    body = sample(np.zeros(3), BODY)
    global N_BODY; N_BODY = len(body)
    pts = [body]
    for c, a in WINGS:
        w = sample(c, a); pts.append(w[(((w) / BODY) ** 2).sum(1) > 1])
    p = np.vstack(pts)
    return p + np.array([NG * DX / 2, FLOOR + BODY[1], NG * DX / 2]), 1.0 / density


P_INIT, VOL_P = make_particles()
N = len(P_INIT)
RHO = MASS / (N * VOL_P)        # g/mm^3, so the particle bee weighs exactly MASS
x = ti.Vector.field(3, ti.f32, N); v = ti.Vector.field(3, ti.f32, N)
C = ti.Matrix.field(3, 3, ti.f32, N); F = ti.Matrix.field(3, 3, ti.f32, N); cmax = ti.field(ti.f32, N)
ks = ti.field(ti.f32, N)        # per-particle stiffness factor (height gradient)
gv = ti.Vector.field(3, ti.f32, (NG, NG, NG)); gm = ti.field(ti.f32, (NG, NG, NG))
mu0 = ti.field(ti.f32, ()); lam0 = ti.field(ti.f32, ()); hh = ti.field(ti.f32, ()); dtf = ti.field(ti.f32, ())
f_floor = ti.field(ti.f32, ()); f_push = ti.field(ti.f32, ()); muf = ti.field(ti.f32, ()); dampf = ti.field(ti.f32, ())
msf = ti.field(ti.f32, ()); hexp = ti.field(ti.i32, ())        # mass-scaling factor, hardening law flag
SCAP = 60.0                     # cap on exponential hardening (stiffness factor)


@ti.kernel
def reset(p0: ti.types.ndarray()):
    for i in range(N):
        x[i] = ti.Vector([p0[i, 0], p0[i, 1], p0[i, 2]]); v[i] = ti.Vector([0.0, 0.0, 0.0])
        C[i] = ti.Matrix.zero(ti.f32, 3, 3); F[i] = ti.Matrix.identity(ti.f32, 3); cmax[i] = 0.0


@ti.kernel
def substep(y_p: ti.f32, v_p: ti.f32):
    dt = dtf[None]
    for I in ti.grouped(gm):
        gv[I] = ti.Vector([0.0, 0.0, 0.0]); gm[I] = 0.0
    for p in range(N):
        Xp = x[p] * INV; base = ti.cast(Xp - 0.5, ti.i32); fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(ti.f32, 3) + dt * C[p]) @ F[p]
        J = F[p].determinant()
        cmax[p] = ti.max(cmax[p], ti.min(ti.max(1.0 - J, 0.0), 0.8))
        hard = 1.0 + hh[None] * cmax[p]                                     # linear compaction hardening
        if hexp[None] > 0: hard = ti.min(ti.exp(hh[None] * cmax[p]), SCAP)  # exponential (snow-like, Stomakhin et al. 2013)
        s = ks[p] * hard; mu = mu0[None] * s; lam = lam0[None] * s
        U, sig, V = ti.svd(F[p]); R = U @ V.transpose()
        Jc = ti.max(J, 0.2)
        kirch = 2 * mu * (F[p] - R) @ F[p].transpose() + ti.Matrix.identity(ti.f32, 3) * lam * (Jc - 1) * Jc
        mass = RHO * VOL_P * msf[None]
        aff =(-dt * VOL_P * 4 * INV * INV) * kirch + mass * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            off = ti.Vector([i, j, k]); dpos = (off.cast(ti.f32) - fx) * DX; wt = w[i][0] * w[j][1] * w[k][2]
            gv[base + off] += wt * (mass * v[p] + aff @ dpos); gm[base + off] += wt * mass
    for I in ti.grouped(gm):
        m = gm[I]
        if m > 0:
            vel = gv[I] / m * ti.exp(-dt * dampf[None]); y = I[1] * DX
            vel[1] -= dt * GRAV / msf[None]          # mass scaling keeps the weight at exactly MASS
            if y <= FLOOR and vel[1] < 0:                                   # floor: only when pressing into it; slides with Coulomb friction, free to separate
                vn = vel[1]; vtf = ti.Vector([vel[0], 0.0, vel[2]]); vtn = vtf.norm(1e-9)
                f_floor[None] += m * (0.0 - vn) / dt
                vel = vtf * ti.max(0.0, 1.0 + muf[None] * vn / vtn)
            sb = ti.min(ti.max((y - (y_p - DX)) / DX, 0.0), 1.0)          # soft one-cell-thick pusher plane
            if sb > 0 and vel[1] > v_p:                                     # pusher: only when the material moves into it; Coulomb friction, free to separate
                vrn = vel[1] - v_p; vtp = ti.Vector([vel[0], 0.0, vel[2]]); vtq = vtp.norm(1e-9)
                vc = vtp * ti.max(0.0, 1.0 - muf[None] * vrn / vtq)
                f_push[None] += m * sb * (v_p - vel[1]) / dt
                vel = (1 - sb) * vel + sb * ti.Vector([vc[0], v_p, vc[2]])
            gv[I] = vel
    for p in range(N):
        Xp = x[p] * INV; base = ti.cast(Xp - 0.5, ti.i32); fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        nv = ti.Vector([0.0, 0.0, 0.0]); nC = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            off = ti.Vector([i, j, k]); dpos = off.cast(ti.f32) - fx; wt = w[i][0] * w[j][1] * w[k][2]
            g = gv[base + off]; nv += wt * g; nC += 4 * INV * wt * g.outer_product(dpos)
        v[p] = nv; C[p] = nC
        x[p] = ti.min(ti.max(x[p] + dt * nv, 3 * DX), (NG - 3) * DX)        # keep stencils inside the grid (safety, never active in a healthy run)


def run(E, nu, hh_, e0, t, e, g, tc=100.0, dt=1.5e-5, lift=8.0, save_every=None, snap_n=0, mu_fric=0.3, snap_cb=None, r_grad=1.0, hard="lin"):
    """simulate the whole sequence; return (floor force gf, pusher force gf) at each sample time (+ optional particle snapshots).
    pusher penetration below the rest crown = (e - e0)*H0 on loaded frames (g > 0), `lift` mm above the crown on 0 g frames.
    E is the modulus at the crown; stiffness grows smoothly toward the bottom: E(y) = E * r_grad**(1 - y/H0) (r_grad = 1: uniform),
    because the video shows the head compressing first and the lower body much less (src/track_face.py)."""
    mu0[None] = E / (2 * (1 + nu)); lam0[None] = E * nu / ((1 + nu) * (1 - 2 * nu)); hh[None] = hh_; dtf[None] = dt; muf[None] = mu_fric
    reset(P_INIT.astype(np.float32))
    yrel = np.clip((P_INIT[:, 1] - FLOOR) / H0, 0.0, 1.0)
    ks.from_numpy((r_grad ** (1.0 - yrel)).astype(np.float32))
    hexp[None] = 1 if hard == "exp" else 0
    E_max = E * max(r_grad, 1.0) * (min(np.exp(hh_ * 0.8), SCAP) if hard == "exp" else 1 + hh_ * 0.8)   # stiffest material in the run
    c_max = np.sqrt(E_max * (1 - nu) / ((1 + nu) * (1 - 2 * nu)) / RHO)
    # CFL needs c*dt < 0.3 DX. Keep dt and, if needed, scale the mass up instead (quasi-static mass scaling: the press is
    # slow, inertia is irrelevant; gravity is scaled down by the same factor so the bee still weighs MASS)
    msf[None] = max(1.0, (c_max * dt / (0.3 * DX)) ** 2)
    global LAST_MASS_SCALE; LAST_MASS_SCALE = float(msf[None])
    dampf[None] = DAMP                                     # settle with the lighter damping: under DAMP_FREE the gravity sag
    ns = int(0.15 / dt)                                    # relaxes in ~0.2 s and would still be sinking. Pusher well clear.
    for s in range(ns):
        if s == int(0.75 * ns): f_floor[None] = 0.0
        substep(FLOOR + H0 + 20.0, 0.0)
    global LAST_SETTLE
    xs = x.to_numpy(); c = np.array([NG * DX / 2, 0, NG * DX / 2])
    core = (np.abs(xs[:, 0] - c[0]) < 6) & (np.abs(xs[:, 2] - c[2]) < 6) & (xs[:, 1] < FLOOR + H0 + 1)
    top = np.percentile(xs[core, 1], 99)                 # settled crown
    LAST_SETTLE = dict(weight_gf=f_floor[None] / (ns - int(0.75 * ns)) / GF, sag_mm=FLOOR + H0 - top)
    pen = np.where(g > 0, (e - e0) * H0, -lift); y = top - pen
    out_f, out_p, snaps = [0.0], [0.0], []
    # pusher speed capped at VMAX (Mach ~0.03 for the softest fits): fast real-time moves get more sim time, which is
    # harmless for a rate-independent material and keeps the loading quasi-static
    nseg = [max(int((t[i + 1] - t[i]) / tc / dt), 20, int(abs(y[i + 1] - y[i]) / (VMAX * dt)) + 1) for i in range(len(t) - 1)]
    K = np.concatenate([[0], np.cumsum(nseg)]).astype(float)
    # smooth pusher path (PCHIP: monotone, no overshoot, continuous velocity). Piecewise-linear motion jumps in speed at
    # every sample and makes the soft body ring ("wiggle"); a C1 path does not excite it.
    from scipy.interpolate import PchipInterpolator
    Ysp = PchipInterpolator(K, y); dYsp = Ysp.derivative()
    for i in range(len(t) - 1):
        n = nseg[i]; kk = K[i] + np.arange(n); yy = Ysp(kk); vv = dYsp(kk) / dt
        dampf[None] = DAMP_FREE if g[i + 1] <= 0 else DAMP
        for s in range(n):
            if s == int(0.75 * n): f_floor[None] = 0.0; f_push[None] = 0.0
            substep(float(yy[s]), float(vv[s]))
            if snap_n and s % max(n // snap_n, 1) == 0:
                snap = (t[i] + (t[i + 1] - t[i]) * s / n, x.to_numpy()[::2], cmax.to_numpy()[::2], float(yy[s]))
                if snap_cb: snap_cb(*snap)            # stream to a renderer instead of keeping every snapshot in memory
                else: snaps.append(snap)
        cnt = n - int(0.75 * n)
        out_f.append(f_floor[None] / cnt / GF - MASS); out_p.append(-f_push[None] / cnt / GF)     # tare: minus the bee's weight
        if save_every and i % save_every == 0: snaps.append((t[i + 1], x.to_numpy()[::6].copy(), float(y[i + 1])))
    return (np.array(out_f), np.array(out_p), snaps) if (save_every or snap_n) else (np.array(out_f), np.array(out_p))
