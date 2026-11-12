"""Fit the 3-D MPM bee to the stepped-hold clip. Free: hh (compaction stiffening), e0 (contact offset); E0 is solved in closed form
(force is linear in E at fixed shape). Held-out cycles are simulated (history matters) but excluded from the loss.
usage: python src/fit_mpm.py   -> out/fit3d_grid.csv, out/fit3d_runs.npz
"""
import sys, itertools, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M

HOLD = {3, 7, 10}; NU = 0.2; E_REF = 5e4; TC = 200.0
d = np.genfromtxt("out/steps_pairs.csv", delimiter=",", names=True, dtype=None, encoding=None)
t, g, conf, e = d["t_s"], d["grams"], d["conf"], d["strain"]
cyc = np.cumsum(g == 0)
use = (g > 0) & (conf != "low")
train = use & ~np.isin(cyc, list(HOLD)); test = use & np.isin(cyc, list(HOLD))
sig = 0.08 * g + 25

def score(f):
    w = 1 / sig[train] ** 2
    s = (w * f[train] * g[train]).sum() / (w * f[train] ** 2).sum()
    r = lambda m: (s * f[m] - g[m]) / g[m]
    loss = (((s * f[train] - g[train]) / sig[train]) ** 2).mean()
    return s, loss, r(train), r(test)

if __name__ == "__main__":
    rows, runs = [], {}
    for hh, e0 in itertools.product([0, 3, 6], [-0.16, -0.12, -0.08, -0.04]):
        f, _ = M.run(E_REF, NU, hh, e0, t, e, g, tc=TC)
        s, loss, rt, rh = score(f); runs[f"{hh}_{e0}"] = f
        rows.append((hh, e0, s * E_REF, loss, np.median(abs(rt)), np.median(abs(rh)), np.sqrt(np.mean(rh ** 2))))
        print(f"hh={hh:>3} e0={e0:.2f}  E0={s*E_REF/1e3:6.1f} kPa  train chi2/pt={loss:6.2f}  train med|err|={100*np.median(abs(rt)):4.1f}%  held-out med|err|={100*np.median(abs(rh)):4.1f}% rms={100*np.sqrt(np.mean(rh**2)):4.1f}%", flush=True)
    np.savetxt("out/fit3d_grid.csv", rows, delimiter=",", header="hh,e0,E0_Pa,train_chi2,train_med_err,heldout_med_err,heldout_rms", comments="")
    np.savez("out/fit3d_runs.npz", **runs)
