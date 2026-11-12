"""Weakly-constrained stage-1 fit: saturating-elastic + plastic model, robust loss.
F_el(e) = K*e/(1+b*e);  yield when F_el > Fy + H*ep.   usage: python src/fit_nl.py [lag_s]
"""
import sys
import numpy as np
from scipy.optimize import least_squares, brentq
sys.path.insert(0, "src")
import fit_lumped as L

def simulate(eps, K, b, Fy, H):
    g = lambda e: K * e / (1 + b * e)
    ep, F, EP = 0.0, np.zeros_like(eps), np.zeros_like(eps)
    for i, e in enumerate(eps):
        f = g(max(e - ep, 0.0))
        if f > Fy + H * ep:
            ep = brentq(lambda x: g(e - x) - Fy - H * x, ep, e)   # monotone: root exists between ep and e
            f = g(e - ep)
        F[i], EP[i] = f, ep
    return F, EP

def resid(p, skip=None, verbose=False):
    K, b, Fy, H = np.exp(p)
    F5, EP5 = simulate(L.e5, K, b, Fy, H); Fl, EPl = simulate(L.el, K, b, Fy, H)
    r, nm = [], []
    for n, (ts, g) in enumerate(L.samples):
        if n == skip: continue
        pr = np.interp(ts - L.LAG, L.t5, F5); r.append((pr - g) / (L.SIG_F * g)); nm.append(f"force@{ts:.1f}s read {g:.0f} pred {pr:.0f}")
    pr = np.interp(15, L.tl, Fl); r.append((pr - 900) / 50); nm.append(f"hold read 900 pred {pr:.0f}")
    for ts, s in L.sets:
        pr = np.interp(ts, L.t5, EP5); r.append((pr - s) / L.SIG_SET); nm.append(f"set@{ts:.1f}s meas {s:.3f} pred {pr:.3f}")
    r.append((EPl[-1] - 0.03) / L.SIG_SET); nm.append(f"set after long hold meas 0.030 pred {EPl[-1]:.3f}")
    if verbose:
        for a, v in zip(nm, r): print(f"  {a:42s} z={v:+.1f}")
    return np.array(r)

def fit(skip=None):
    best = None
    for k0 in (3000, 8000):
        for b0 in (1, 8):
            for fy0 in (400, 800):
                s = least_squares(resid, np.log([k0, b0, fy0, 1000]), kwargs={"skip": skip}, loss="soft_l1", f_scale=2.0,
                                  bounds=(np.log([100, 1e-3, 50, 1]), np.log([1e6, 1e3, 3000, 1e6])))
                if best is None or s.cost < best.cost: best = s
    return best

if __name__ == "__main__":
    s = fit(); K, b, Fy, H = np.exp(s.x)
    print(f"lag {L.LAG}s  K={K:.0f} g/strain  b={b:.2f}  Fy={Fy:.0f} g  H={H:.0f} g/strain  cost={s.cost:.1f}")
    resid(s.x, verbose=True)
    print("leave-one-out:")
    for n, (ts, g) in enumerate(L.samples):
        sk = fit(skip=n); F5, _ = simulate(L.e5, *np.exp(sk.x)); pr = np.interp(ts - L.LAG, L.t5, F5)
        print(f"  held out {g:.0f} g -> predicted {pr:.0f} g ({100*(pr-g)/g:+.0f}%)")
    np.save("out/fit_nl_params.npy", np.exp(s.x))
