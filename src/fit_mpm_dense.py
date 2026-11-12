"""Fit the 3-D MPM bee (gravity on, 10 g, tared) to the PER-FRAME data. The pusher follows the cleaned 60 fps strain,
sampled every 0.25 s (the display's update rate); every stable display value is a target, shifted by the display lag
fitted in fit_dense.py. Free: hh, e0 (grid); E0 by closed-form scaling, then the best point is re-run at that E0 so the
reported errors are from a real simulation, not the scaling. Held-out cycles excluded from the loss.
usage: python src/fit_mpm_dense.py  -> out/fit3d_dense_grid.csv, out/fit3d_dense_best.json, out/fit3d_dense_runs.npz
"""
import sys, json, itertools, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M
import fit_dense as D

NU, E_REF, TC, LAG = 0.2, 1.2e4, 50.0, 0.21    # TC: time compression (see mpm_bee.DAMP); nu = 0.2: matches the eye-marker deformation better than 0 (face_sim.py)
step = 15                                                   # 0.25 s
# the measured strain has frame-level noise (std ~0.3 mm, spikes ~1.6 mm, from edge detection); the hand-held pusher
# moves smoothly, so low-pass it (zero-phase 2nd-order Butterworth, 2 Hz; presses ramp over 1-2 s) before it drives the
# pusher -- unfiltered, the simulated bee bobs with the noise
from scipy.signal import butter, filtfilt
_b, _a = butter(2, 2.0 / 30.0)
# The pusher path and the contact flag come from the VIDEO (the measured pusher edge on every frame), not from the scale
# display: the display lags ~0.2 s and drops to 0 up to ~1 s before the stone actually leaves the bee, which made the
# simulated bee spring back early at the end of heavy presses. When the stone is off, this measure reads the top of the
# wings (~ -0.09), so contact = strain above -0.03 (kept on for 0.1 s across short dips).
from scipy.signal import medfilt
from scipy.ndimage import binary_closing
_fr = np.genfromtxt("out/frames.csv", delimiter=",", names=True)
_tl = np.nan_to_num(_fr["top_loose"], nan=np.nanmedian(_fr["top_loose"])); _bl = np.nan_to_num(_fr["bot_loose"], nan=np.nanmedian(_fr["bot_loose"]))
E_RAW = 1 - (_bl - _tl) / 320.0
E_VIDEO = medfilt(E_RAW, 9)
E_SMOOTH = filtfilt(_b, _a, E_VIDEO)
CONTACT = binary_closing(E_SMOOTH > -0.03, np.ones(7, bool))
ts = D.t[::step]; es = E_SMOOTH[::step]; ls = CONTACT[::step].astype(float)
allc = set(np.unique(D.cyc_of_run)); tr, te = D.targets(allc - D.HOLD), D.targets(D.HOLD)

def at_targets(f, tg): return np.interp(D.t[tg[:, 0].astype(int)] - LAG, ts, f)

def score(f):
    a, b = at_targets(f, tr), tr[:, 2]; w = 1 / (0.03 * b + 10) ** 2
    s = (w * a * b).sum() / (w * a * a).sum()
    return s, ((s * a - b) ** 2 * w).mean()

def errs(f, tg): return (at_targets(f, tg) - tg[:, 2]) / tg[:, 2]

def _grid_run(hh, e0): return M.run(E_REF, NU, hh, e0, ts, es, ls, tc=TC)[0]

if __name__ == "__main__":
    from multiprocessing import Pool
    rows, runs = [], {}
    grid = list(itertools.product([3, 6, 10], [-0.12, -0.08, -0.04, 0.0]))
    with Pool(3) as pool:                                  # 3 sims share the GPU (~3x faster than in sequence)
        fs = pool.starmap(_grid_run, grid)
    for (hh, e0), f in zip(grid, fs):
        s, loss = score(f); runs[f"{hh}_{e0}"] = f
        eh = errs(s * f, te); et = errs(s * f, tr)
        rows.append((hh, e0, s * E_REF, loss, np.median(abs(et)), np.median(abs(eh)), np.sqrt(np.mean(eh ** 2))))
        print(f"hh={hh:>3} e0={e0:+.2f} E0~{s*E_REF/1e3:5.1f} kPa  train chi2={loss:6.2f} med {100*np.median(abs(et)):4.1f}% | held-out med {100*np.median(abs(eh)):4.1f}% rms {100*np.sqrt(np.mean(eh**2)):4.1f}%", flush=True)
    rows = np.array(rows); b = rows[np.argmin(rows[:, 3])]
    hh, e0, E0 = b[0], b[1], b[2]
    f, _ = M.run(E0, NU, hh, e0, ts, es, ls, tc=TC); et, eh = errs(f, tr), errs(f, te)        # real re-run at the fitted E0
    best = dict(hh=float(hh), e0=float(e0), E0_Pa=float(E0), nu=NU, lag_s=LAG, settle=M.LAST_SETTLE, held_out_cycles=sorted(D.HOLD),
                train_med=float(np.median(abs(et))), heldout_med=float(np.median(abs(eh))), heldout_rms=float(np.sqrt(np.mean(eh ** 2))), heldout_max=float(abs(eh).max()))
    print("BEST (re-run at fitted E0):", json.dumps(best, default=float))
    np.savetxt("out/fit3d_dense_grid.csv", rows, delimiter=",", header="hh,e0,E0_Pa,train_chi2,train_med_err,heldout_med_err,heldout_rms", comments="")
    json.dump(best, open("out/fit3d_dense_best.json", "w"), default=float)
    np.savez("out/fit3d_dense_runs.npz", ts=ts, best=f, **runs)
