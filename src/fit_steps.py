"""Fit force-vs-strain models on the stabilized stepped-hold clip; validate by leaving out whole press cycles.
Strain (video) drives the model, display force is the target. Low-confidence display reads are not fitted (their strain is still used).
Models: A linear+offset | B +quadratic stiffening | C = B + Maxwell branch (stress relaxation, tau).
usage: python src/fit_steps.py
"""
import numpy as np
from scipy.optimize import least_squares
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

d = np.genfromtxt("out/steps_pairs.csv", delimiter=",", names=True, dtype=None, encoding=None)
t, g, conf, e = d["t_s"], d["grams"], d["conf"], d["strain"]
cyc = np.cumsum(g == 0)                         # new cycle starts at each released (0 g) frame
cycles = []
for c in np.unique(cyc):
    i = np.where((cyc == c))[0]                 # 0 g start frame + loaded frames until next release
    if len(i) > 2: cycles.append(i)

def sim(p, i, model):
    """force prediction for the frames in index list i (first = release frame, state starts relaxed there)."""
    e0, K = p[0], p[1]; K2 = p[2] if model in "BC" else 0.0
    h = p[2] if model == "D" else 0.0
    K1, tau = (p[3], p[4]) if model == "C" else (0.0, 1.0)
    F, q, ep = [], 0.0, e[i[0]]
    for k, j in enumerate(i):
        x = max(e[j] - e0, 0.0) * (1 + h * max(e[:j + 1].max(), 0.0))   # D: stiffness grows with peak strain reached so far
        if k > 0:
            dt = t[j] - t[i[k - 1]]; de = e[j] - e[i[k - 1]]
            q = q * np.exp(-dt / tau) + K1 * de * (tau / dt) * (1 - np.exp(-dt / tau))   # Maxwell branch, exact for linear strain ramp
        F.append(max(K * x + K2 * x * x + q, 0.0))
    return np.array(F)

def resid(p, idx, model):
    r = []
    for i in idx:
        f = sim(np.exp(p) if False else p, i, model)
        for k, j in enumerate(i):
            if k > 0 and conf[j] != "low": r.append((f[k] - g[j]) / (0.08 * g[j] + 25))
    return np.array(r)

X0 = {"D": [0.05, 3000, 3.0], "A": [0.05, 4000], "B": [0.05, 3000, 3000], "C": [0.05, 3000, 3000, 1500, 3.0]}
LO = {"D": [-0.1, 100, 0], "A": [-0.1, 100], "B": [-0.1, 100, 0], "C": [-0.1, 100, 0, 0, 0.3]}
HI = {"D": [0.3, 5e4, 50], "A": [0.3, 5e4], "B": [0.3, 5e4, 5e4], "C": [0.3, 5e4, 5e4, 5e4, 60]}

def fit(idx, model):
    return least_squares(resid, X0[model], bounds=(LO[model], HI[model]), args=(idx, model), loss="soft_l1", f_scale=1.5).x

if __name__ == "__main__":
    print(f"{len(cycles)} cycles, held-out error = leave-one-cycle-out, relative to the display read (excl. low-confidence)")
    res = {}
    for m in "ABCD":
        p = fit(cycles, m); errs = []
        for h in range(len(cycles)):
            ph = fit([c for k, c in enumerate(cycles) if k != h], m); f = sim(ph, cycles[h], m)
            for k, j in enumerate(cycles[h]):
                if k > 0 and conf[j] != "low": errs.append((f[k] - g[j]) / g[j])
        errs = np.array(errs); res[m] = p
        print(f"model {m}: params {np.round(p, 4)}  LOO median|err| {100*np.median(abs(errs)):.1f}%  rms {100*np.sqrt(np.mean(errs**2)):.1f}%  worst {100*abs(errs).max():.0f}%")
    p = res["C"]; fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t, g, "ko", ms=3, label="display read"); ax[0].plot(t[conf == "low"], g[conf == "low"], "rx", label="low conf")
    for i in cycles: ax[0].plot(t[i], sim(p, i, "C"), "-", c="tab:blue")
    ax[0].plot([], [], c="tab:blue", label="model C (all data)"); ax[0].set_ylabel("force (g)"); ax[0].legend()
    ax[1].plot(t, e, "o-", ms=3); ax[1].set_ylabel("strain (video)"); ax[1].set_xlabel("t (s)")
    fig.savefig("out/fit_steps.png", dpi=110, bbox_inches="tight")
