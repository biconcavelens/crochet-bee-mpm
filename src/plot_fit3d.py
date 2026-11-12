"""Results figure for the fitted 3-D MPM bee: simulated vs displayed force over the whole clip (held-out cycles shaded)
and the per-cycle median error. Reads the saved best run (fit_mpm_dense.py), no simulation.
usage: python src/plot_fit3d.py  -> out/fit3d_dense.png  (+ per-cycle table on stdout)
"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "src")
import fit_dense as D
from fit_mpm_dense import ts, LAG, errs, tr, te

f = np.load("out/fit3d_dense_runs.npz")["best"]; b = json.load(open("out/fit3d_dense_best.json"))
tg = np.vstack([tr, te]); cyc = D.cyc_of_run[np.searchsorted(D.runs[:, 0], tg[:, 0])]
e = errs(f, tg)
fig, (a, c) = plt.subplots(2, 1, figsize=(13, 7), gridspec_kw=dict(height_ratios=[2, 1]))
for k in sorted(D.HOLD):
    r = D.runs[D.cyc_of_run == k]; a.axvspan(D.t[int(r[0, 0])], D.t[int(r[-1, 1])], color="#fdd", lw=0)
a.plot(D.t[tg[:, 0].astype(int)] - LAG, tg[:, 2], "k.", ms=3, label="scale display (lag-corrected)")
a.plot(ts, np.maximum(f, 0), color="C1", lw=1.2, label="3-D MPM simulation")
a.set(xlabel="time (s)", ylabel="force (g)", title=f"E0 = {b['E0_Pa']/1e3:.2f} kPa, hh = {b['hh']:g}, e0 = {b['e0']:+.2f}, nu = {b['nu']}"
      f"   |   held-out (pink): median {100*b['heldout_med']:.1f}%, rms {100*b['heldout_rms']:.1f}%")
a.legend(loc="upper left")
ks = np.unique(cyc); med = [100 * np.median(abs(e[cyc == k])) for k in ks]
c.bar(ks, med, color=["#e66" if k in D.HOLD else "0.6" for k in ks])
c.set(xlabel="press cycle (red = held out, never used for fitting)", ylabel="median |error| (%)", xticks=ks)
plt.tight_layout(); plt.savefig("out/fit3d_dense.png", dpi=110)
for k, m in zip(ks, med): print(f"cycle {int(k):2d} {'HELD-OUT' if k in D.HOLD else 'train   '} median |err| {m:5.1f}%  n={int((cyc == k).sum())}")
