"""Screen material variants on BOTH observables: how the bee deforms (eye markers, face_sim.py) and the force (train
cycles, closed-form E). usage: python src/screen_shape.py  -> out/screen_shape2.csv"""
import sys, itertools, numpy as np
sys.path.insert(0, "src")
import mpm_bee as M
from face_sim import face_metrics, real_metrics
from fit_mpm_dense import score, errs, tr, te, E_REF, ls

real = real_metrics(); rows = []
for xi, r in itertools.product([4.0, 7.0], [1.0, 10.0]):
    fo = []; sim = face_metrics(E_REF, 0.0, xi, 0.0, r_grad=r, f_out=fo, hard="exp"); f, p = fo[0]
    s, loss = score(f); eh = errs(s * f, te)
    m = ls > 0; qs = np.median(abs(f[m] - p[m]) / np.maximum(f[m], 1))
    shape_err = np.sqrt(np.nanmean((sim - real) ** 2, axis=0))
    rows.append((xi, r, s * E_REF, loss, np.median(abs(eh)), shape_err[0], shape_err[1], M.LAST_MASS_SCALE, qs))
    print(f"exp xi={xi} r={r:4.0f}  E_top~{s*E_REF/1e3:5.2f} kPa  mass-scale {M.LAST_MASS_SCALE:5.1f}  floor/pusher mismatch {100*qs:4.1f}% | force chi2 {loss:6.2f} held-out med {100*np.median(abs(eh)):4.1f}% | "
          f"below-eyes {np.round(sim[:, 0], 3).tolist()} (real {np.round(real[:, 0], 3).tolist()})  spacing {np.round(sim[:, 1], 3).tolist()}", flush=True)
np.savetxt("out/screen_shape2.csv", rows, delimiter=",", comments="", header="xi,r,E_top,force_chi2,heldout_med,rms_low,rms_spacing,mass_scale,floor_pusher_mismatch")
print("done")
