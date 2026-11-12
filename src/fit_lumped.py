"""Stage 1: 1-D elasto-plastic model of the toy, driven by the measured strain, fitted to sparse force data.

State: plastic strain ep. F = K*(eps-ep) (grams-force per unit strain), yield when F > Fy + H*ep.
Fit params: K, Fy, H.  Data: 6 clean display reads (ruler clip), 900 g hold (long clip),
residual set between presses (ruler clip) and after release (long clip).
usage: python src/fit_lumped.py [lag_s]
"""
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import medfilt

sys.path.insert(0, "src")
from press_events import events

LAG = float(sys.argv[1]) if len(sys.argv) > 1 else 0.2   # display lags the true force (s)
SIG_F = 0.12   # relative force error: reading + timing uncertainty
SIG_SET = 0.012  # strain error on measured permanent set


def load(csv):
    d = np.genfromtxt(csv, delimiter=",", names=True)
    return d["t_s"], medfilt(d["bottom_px"] - d["top_px"], 5)


def simulate(eps, K, Fy, H):
    ep, F, EP = 0.0, np.zeros_like(eps), np.zeros_like(eps)
    for i, e in enumerate(eps):
        f = K * max(e - ep, 0.0)
        if f > Fy + H * ep:                       # return mapping, linear hardening
            ep += (f - Fy - H * ep) / (K + H)
            f = K * (e - ep)
        F[i], EP[i] = f, ep
    return F, EP


# ---- data -------------------------------------------------------------------------------
t5, h5 = load("out/clip5_track.csv"); h05 = np.median(h5[:6]); e5 = 1 - h5 / h05
_, _, _, pk5 = events("out/clip5_track.csv")
tl, hl = load("out/long_track.csv"); h0l = np.median(hl[tl < 0.4]); el = 1 - hl / h0l

samples = np.genfromtxt("out/clip5_force_samples.csv", delimiter=",", skip_header=1, usecols=(1, 2))  # t, grams
# permanent set between presses = strain at the local top between peaks (skip 8->9: the ruler was held there)
sets = []
for i in range(len(pk5) - 1):
    if i + 1 == 8:
        continue
    seg = np.arange(pk5[i], pk5[i + 1]); j = seg[np.argmax(h5[seg])]
    sets.append((t5[j], 1 - np.median(h5[max(j - 3, 0):j + 4]) / h05))


def residuals(p, verbose=False, skip=None):
    K, Fy, H = np.exp(p)
    F5, EP5 = simulate(e5, K, Fy, H); Fl, EPl = simulate(el, K, Fy, H)
    r, names = [], []
    for n, (ts, g) in enumerate(samples):
        if n == skip:
            continue
        pred = np.interp(ts - LAG, t5, F5); r.append((pred - g) / (SIG_F * g)); names.append(f"force@{ts:.1f}s {g:.0f}g pred {pred:.0f}")
    pred = np.interp(15, tl, Fl); r.append((pred - 900) / 50.0); names.append(f"hold 900g pred {pred:.0f}")
    for ts, s in sets:
        pred = np.interp(ts, t5, EP5); r.append((pred - s) / SIG_SET); names.append(f"set@{ts:.1f}s meas {s:.3f} pred {pred:.3f}")
    pred = EPl[-1]; r.append((pred - 0.03) / SIG_SET); names.append(f"set after long hold meas 0.030 pred {pred:.3f}")
    if verbose:
        for n, v in zip(names, r):
            print(f"  {n:45s} z={v:+.1f}")
    return np.array(r)


def fit(skip=None):
    best = None
    for k0 in (2000, 5000, 15000):
        for fy0 in (300, 800):
            s = least_squares(residuals, np.log([k0, fy0, 3000]), kwargs={"skip": skip}, bounds=(np.log([100, 50, 1]), np.log([1e6, 3000, 1e6])))
            if best is None or s.cost < best.cost:
                best = s
    return best


if __name__ == "__main__":
    s = fit(); K, Fy, H = np.exp(s.x)
    print(f"lag {LAG}s  K={K:.0f} g/strain ({K*9.81e-3:.1f} N/strain)  Fy={Fy:.0f} g ({Fy*9.81e-3:.1f} N)  H={H:.0f} g/strain  yield strain~{Fy/K:.3f}  cost={s.cost:.2f}")
    residuals(s.x, verbose=True)
    # leave-one-out on the six display reads: predict the one left out
    print("leave-one-out force predictions:")
    for n, (ts, g) in enumerate(samples):
        sk = fit(skip=n); F5, _ = simulate(e5, *np.exp(sk.x)); pred = np.interp(ts - LAG, t5, F5)
        print(f"  held out {g:.0f} g at {ts:.1f}s -> predicted {pred:.0f} g ({100*(pred-g)/g:+.0f}%)")
