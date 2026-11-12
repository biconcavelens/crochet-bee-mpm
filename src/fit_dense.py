"""Fit 1-D force models to the PER-FRAME data (out/series.csv): strain at 60 fps drives the model, every stable display
value (~4 Hz) is a target. Validated by leaving out whole press cycles (same held-out cycles as the 3-D fit).

Models (all have contact offset e0, display lag):
  L   linear spring                      F = K (e - e0)+
  D   + history stiffening               F = K (1 + h emax) (e - e0)+          emax = deepest strain so far
  DM  D + one Maxwell branch (SLS)       F = D + q,  dq/dt = K1 (1 + h emax) de/dt - q / tau   (stress relaxation / creep)
usage: python src/fit_dense.py
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import medfilt

HOLD = {3, 7, 10}
d = np.genfromtxt("out/series.csv", delimiter=",", names=True)
t, F_disp, e_raw = d["t_s"], d["force_g"], d["strain"]
n = len(t); dt = t[1] - t[0]

# ---- display runs -> targets: (time the value appeared, value) -----------------------------------------------------
runs = []; k = 0
while k < n:
    if np.isnan(F_disp[k]): k += 1; continue
    j = k
    while j + 1 < n and F_disp[j + 1] == F_disp[k]: j += 1
    runs.append((k, j, F_disp[k])); k = j + 1
runs = np.array(runs)
zero = runs[:, 2] == 0
# press cycles: a cycle starts at each zero-force run that follows a loaded run
cyc_of_run = np.cumsum(np.r_[1, (zero[1:] & ~zero[:-1])]) if len(runs) else []
# ---- strain cleaning: drop spikes and the release/approach transitions, interpolate across them ---------------------
ok_idx = np.where(~np.isnan(F_disp))[0]                                             # fill gaps between display values with the
F_fill = F_disp[ok_idx[np.clip(np.searchsorted(ok_idx, np.arange(n)), 0, len(ok_idx) - 1)]]   # next stable value
loaded_disp = F_fill > 0
spike = np.abs(e_raw - medfilt(e_raw, 31)) > 0.03
edge = np.zeros(n, bool)
for i in np.where(np.diff(loaded_disp.astype(int)) != 0)[0]: edge[max(i - 30, 0):i + 31] = True     # +-0.5 s around contact changes
bad = spike | edge
e = e_raw.copy(); e[bad] = np.interp(np.where(bad)[0], np.where(~bad)[0], e_raw[~bad])
e_hist = np.maximum.accumulate(np.where(loaded_disp & ~bad, e, 0.0))            # deepest clean loaded strain so far


def sim(e, emax, dt, e0, K, h, K1, tau):
    F = np.zeros(len(e)); q = 0.0
    for i in range(len(e)):
        x = e[i] - e0; s = 1.0 + h * emax[i]
        if x <= 0: q = 0.0; F[i] = 0.0; continue
        if i > 0 and tau > 0:
            de = e[i] - e[i - 1]; a = np.exp(-dt / tau)
            q = q * a + K1 * s * de * tau / dt * (1 - a)
        F[i] = max(K * s * x + q, 0.0)
    return F

NAMES = {"L": ["e0", "K", "lag"], "D": ["e0", "K", "h", "lag"], "DM": ["e0", "K", "h", "K1", "tau", "lag"]}
X0 = {"L": [0.0, 3000, 0.25], "D": [0.0, 1500, 3.0, 0.25], "DM": [0.0, 1500, 3.0, 800, 2.0, 0.25]}
LO = {"L": [-0.2, 100, 0.0], "D": [-0.2, 100, 0, 0.0], "DM": [-0.2, 100, 0, 0, 0.05, 0.0]}
HI = {"L": [0.2, 3e4, 0.8], "D": [0.2, 3e4, 50, 0.8], "DM": [0.2, 3e4, 50, 3e4, 60, 0.8]}

def predict(p, m):
    q = dict(zip(NAMES[m], p))
    F = sim(e, e_hist, dt, q["e0"], q["K"], q.get("h", 0.0), q.get("K1", 0.0), q.get("tau", 0.0))
    return F, q["lag"]

def targets(sel_cycles):
    """loaded display runs in the selected cycles, skipping runs that start inside a cleaned-out transition."""
    m = (runs[:, 2] > 0) & np.isin(cyc_of_run, list(sel_cycles))
    m &= ~bad[runs[:, 0].astype(int)]
    return runs[m]

def resid(p, m, tg):
    F, lag = predict(p, m)
    tm = t[tg[:, 0].astype(int)] - lag
    pred = np.interp(tm, t, F)
    return (pred - tg[:, 2]) / (0.03 * tg[:, 2] + 10)        # reader ~1-2% + model; relative-ish weighting

def fit(m, tg):
    return least_squares(resid, X0[m], bounds=(LO[m], HI[m]), args=(m, tg), loss="soft_l1", f_scale=2.0, x_scale="jac").x

if __name__ == "__main__":
    allc = set(np.unique(cyc_of_run)); train = allc - HOLD
    tg_tr, tg_te = targets(train), targets(HOLD)
    print(f"{len(runs)} display values; loaded targets: train {len(tg_tr)}, held-out {len(tg_te)}; strain frames cleaned {bad.mean():.1%}")
    res = {}
    for m in ("L", "D", "DM"):
        p = fit(m, tg_tr); res[m] = p
        F, lag = predict(p, m)
        def err(tg): pr = np.interp(t[tg[:, 0].astype(int)] - lag, t, F); return (pr - tg[:, 2]) / tg[:, 2]
        et, eh = err(tg_tr), err(tg_te)
        print(f"{m:3s} " + " ".join(f"{k}={v:.4g}" for k, v in zip(NAMES[m], p)) +
              f"\n     train med|err| {100*np.median(abs(et)):4.1f}% rms {100*np.sqrt(np.mean(et**2)):4.1f}%   HELD-OUT med|err| {100*np.median(abs(eh)):4.1f}% rms {100*np.sqrt(np.mean(eh**2)):4.1f}% max {100*abs(eh).max():3.0f}%")
    np.savez("out/fit_dense.npz", **{m: v for m, v in res.items()}, e=e, e_hist=e_hist, bad=bad)
