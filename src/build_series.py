"""Per-frame dataset for the stepped-hold clip: display force (temporal-consensus digit reads) + strain (stabilized video).
needs out/frames.csv and cache/lcd.npy from stabilize_all.py, out/lcd_gain.npy from the digit-reader calibration.
usage: python src/build_series.py  -> out/series.csv (frame, t_s, force_raw, force_g, strain)  + out/series.png
"""
import sys, numpy as np
from scipy.signal import medfilt
sys.path.insert(0, "src")
import lcd_read as L

if __name__ == "__main__":
    fr = np.genfromtxt("out/frames.csv", delimiter=",", names=True)
    stack = np.load("cache/lcd.npy")
    gain = np.load("out/lcd_gain.npy")
    raw, conf = L.read_stack(stack, gain)
    force, runs = L.consensus(raw)
    t = fr["t_s"]; n = len(t)
    print(f"frames {n}: decoded {np.mean(raw >= 0):.1%}, in a stable run {np.mean(~np.isnan(force)):.1%}, display values {len(runs)}")
    d = np.diff([a for a, b, v in runs]) / 60.0
    print(f"display update interval: median {np.median(d):.2f} s, 10-90% {np.percentile(d, 10):.2f}-{np.percentile(d, 90):.2f} s")

    # strain: loaded -> pusher edge (top_loose); free (0 g) -> crown. rest reference from the true rest periods (start/end)
    top_l = medfilt(np.nan_to_num(fr["top_loose"], nan=np.nanmedian(fr["top_loose"])), 5)
    bot = medfilt(np.nan_to_num(fr["bot_loose"], nan=np.nanmedian(fr["bot_loose"])), 5)
    top_c = medfilt(np.nan_to_num(fr["top_crown"], nan=np.nanmedian(fr["top_crown"])), 5)
    rest = ((t < 1.5) | (t > 188.5)) & (force == 0)
    hr = np.median(bot[rest] - top_c[rest])
    ff = force.copy()                                   # fill transitions with the nearest stable value, for the loaded/free choice only
    idx = np.where(~np.isnan(ff))[0]; ff = ff[idx[np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)]]
    loaded = ff > 0
    strain = np.where(loaded, 1 - (bot - top_l) / hr, 1 - (bot - top_c) / hr)
    print(f"rest height {hr:.1f} px from {rest.sum()} rest frames; strain at rest frames: {np.median(strain[rest]):+.4f} +- {np.std(strain[rest]):.4f}")
    np.savetxt("out/series.csv", np.column_stack([fr["frame"], t, raw, force, strain]), delimiter=",",
               header="frame,t_s,force_raw,force_g,strain", comments="", fmt=["%d", "%.4f", "%d", "%.1f", "%.4f"])

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(t, np.where(raw >= 0, raw, np.nan), ".", ms=1, c="0.7", label="single-frame read")
    ax[0].plot(t, force, "-", c="k", lw=1, label="stable (consensus) read")
    man = np.genfromtxt("out/steps_force_reads.csv", delimiter=",", names=True, dtype=None, encoding=None)
    ax[0].plot(man["t_s"], man["grams"], "o", mfc="none", c="tab:red", label="manual read (3 s)")
    ax[0].set_ylabel("display (g)"); ax[0].legend(loc="upper left")
    ax[1].plot(t, strain, lw=0.8); ax[1].set_ylabel("strain")
    ok = ~np.isnan(force) & loaded
    ax[2].plot(t[ok], force[ok] / np.maximum(strain[ok], 0.02), ".", ms=1); ax[2].set_ylabel("force/strain (g)"); ax[2].set_xlabel("t (s)")
    fig.savefig("out/series.png", dpi=110, bbox_inches="tight")
