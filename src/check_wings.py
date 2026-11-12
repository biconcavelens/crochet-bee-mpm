"""Integrity check of the simulated bee over the whole clip: does any material (e.g. a wing) tear off?
Builds a neighbour graph (particles closer than R mm) at every snapshot and counts particles outside the largest
connected piece. Run with the fitted parameters.  usage: python src/check_wings.py
"""
import sys, json, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "src")
import mpm_bee as M
from fit_mpm_dense import ts, es, ls, TC

R = 2.5            # mm; snapshots keep every 2nd particle (mean spacing ~1.4 mm), so 2.5 mm links true neighbours only
b = json.load(open("out/fit3d_dense_best.json"))
wing = (np.arange(M.N) >= M.N_BODY)[::2]
worst = []

def cb(tm, x, cm, yp):
    G = cKDTree(x).sparse_distance_matrix(cKDTree(x), R, output_type="coo_matrix")
    n, lab = connected_components(G, directed=False)
    main = np.bincount(lab).argmax(); off = lab != main
    worst.append((tm, int(off.sum()), int((off & wing).sum()), yp))

M.run(b["E0_Pa"], b["nu"], b["hh"], b["e0"], ts, es, ls, tc=TC, snap_n=1, snap_cb=cb)
w = np.array(worst)
print(f"{len(w)} snapshots checked; max particles detached from the main body: {int(w[:, 1].max())} (of which wing: {int(w[:, 2].max())})")
bad = w[w[:, 1] > 0]
if len(bad): print("first detachment at t=%.1f s, %d particles" % (bad[0, 0], bad[0, 1]))
