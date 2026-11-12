"""Compare 3-D MPM vs 1-D model D on the same held-out cycles; plot the best run. usage: python src/report_mpm.py"""
import sys, json, numpy as np
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from fit_mpm import t, g, conf, e, cyc, train, test, score, E_REF, HOLD
import fit_steps as S

G = np.genfromtxt("out/fit3d_grid.csv", delimiter=",", names=True)
best = G[np.argmin(G["train_chi2"])]; key = f"{int(best['hh'])}_{best['e0']}"
f = np.load("out/fit3d_runs.npz")[key]; s, loss, rt, rh = score(f); pred = s * f
print("3-D best by TRAIN loss: hh=%g e0=%g E0=%.1f kPa | train med %.1f%% | held-out med %.1f%% rms %.1f%% max %.0f%%" % (
    best["hh"], best["e0"], s * E_REF / 1e3, 100 * np.median(abs(rt)), 100 * np.median(abs(rh)), 100 * np.sqrt(np.mean(rh ** 2)), 100 * abs(rh).max()))
near = G[G["train_chi2"] < 1.1 * best["train_chi2"]]
print("runs within 10%% of best train loss: %d  E0 range %.1f-%.1f kPa, hh %g-%g, e0 %g to %g" % (len(near), near["E0_Pa"].min() / 1e3, near["E0_Pa"].max() / 1e3, near["hh"].min(), near["hh"].max(), near["e0"].min(), near["e0"].max()))

ids = [k for k, c in enumerate(S.cycles)]; hold_idx = [k for k in ids if (k + 1) in HOLD]
p1 = S.fit([S.cycles[k] for k in ids if k not in hold_idx], "D"); errs = []
for k in hold_idx:
    fp = S.sim(p1, S.cycles[k], "D")
    for q, j in enumerate(S.cycles[k]):
        if q > 0 and conf[j] != "low": errs.append((fp[q] - g[j]) / g[j])
errs = np.array(errs)
print("1-D model D, same split: held-out med %.1f%% rms %.1f%% max %.0f%%" % (100 * np.median(abs(errs)), 100 * np.sqrt(np.mean(errs ** 2)), 100 * abs(errs).max()))

fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].plot(t, pred, "-", c="tab:blue", label="3-D MPM (simulated floor force)")
ax[0].plot(t[train], g[train], "ko", ms=4, label="display read (train)"); ax[0].plot(t[test], g[test], "o", c="tab:red", ms=5, label="display read (HELD OUT cycles)")
ax[0].plot(t[conf == "low"], g[conf == "low"], "x", c="gray", label="low-confidence read")
ax[0].set_ylabel("force (g)"); ax[0].legend(loc="upper left"); ax[1].plot(t, e, "o-", ms=3); ax[1].set_ylabel("strain (input)"); ax[1].set_xlabel("t (s)")
fig.savefig("out/fit3d.png", dpi=110, bbox_inches="tight")
json.dump(dict(hh=best["hh"], e0=best["e0"], E0_Pa=float(s * E_REF), nu=0.2, held_out_cycles=sorted(HOLD)), open("out/fit3d_best.json", "w"))
