"""Read the scale's 7-segment display from the stabilized grayscale crops (cache/lcd.npy, made by stabilize_all.py).

1. Register: find the bright LCD rectangle in the crop (largest bright blob) and map its right/top/bottom edges to a fixed
   normalized frame (the left edge is sometimes outside the camera frame, so it is not used).
2. Sample the 7 segments of each of 4 digit cells (italic: segment points are sheared) as the mean darkness of a small patch
   relative to the local LCD background.
3. Decode: a segment is ON if its darkness is above a per-frame threshold halfway between the dark (lit segment) and bright
   (background/ghost) clusters. Patterns that are not valid digits -> unreadable (display refresh, motion blur, occlusion).
"""
import cv2, numpy as np

NW, NH = 400, 160          # normalized LCD frame: right edge at x=NW, top at y=0, bottom at y=NH

def register(c):
    """c: grayscale crop. returns the crop warped so the LCD's right/top/bottom edges are at x=NW, y=0, y=NH, or None."""
    b = cv2.GaussianBlur(c, (5, 5), 0)
    thr = np.percentile(b, 75) * 0.75
    m = (b > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    i = 1 + np.argmax(st[1:, 4]); ys, xs = np.where(lab == i)
    # robust edges: top/bottom per column (median over central columns), right edge per row (median over central rows)
    cols = np.unique(xs); cols = cols[(cols > np.percentile(cols, 40)) & (cols < np.percentile(cols, 95))]
    tops = np.array([ys[xs == x].min() for x in cols[::3]]); bots = np.array([ys[xs == x].max() for x in cols[::3]])
    rws = np.unique(ys); rws = rws[(rws > np.percentile(rws, 25)) & (rws < np.percentile(rws, 75))]
    rights = np.array([xs[ys == y].max() for y in rws[::3]])
    # fit lines (the display can be slightly rotated in the stabilized view)
    pt = np.polyfit(cols[::3], tops, 1); pb = np.polyfit(cols[::3], bots, 1); pr = np.polyfit(rws[::3], rights, 1)
    def right_x(y): return np.polyval(pr, y)
    yt_r = np.polyval(pt, right_x(np.polyval(pt, cols.max()))); yb_r = np.polyval(pb, right_x(np.polyval(pb, cols.max())))
    xr_t, xr_b = right_x(yt_r), right_x(yb_r)
    w = xr_t - (cols.min() - 5)
    src = np.float32([[xr_t, yt_r], [xr_b, yb_r], [xr_t - 0.8 * w, np.polyval(pt, xr_t - 0.8 * w)]])
    dst = np.float32([[NW, 0], [NW, NH], [NW - 0.8 * w * NH / (yb_r - yt_r), 0]])
    A = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(c, A, (NW + 20, NH), borderValue=0)


SEG = {"a": (0, -57, "h"), "b": (23, -29, "v"), "c": (23, 31, "v"), "d": (0, 60, "h"), "e": (-23, 31, "v"), "f": (-23, -29, "v"), "g": (0, 0, "h")}
DIG = {"abcdef": 0, "bc": 1, "abdeg": 2, "abcdg": 3, "bcfg": 4, "acdfg": 5, "acdefg": 6, "abc": 7, "abcdefg": 8, "abcdfg": 9, "": None,
       "acdefg ": 6, "abcf": 7, "cdefg": 6, "abcfg": 9}          # also accept 6/7/9 variants some LCD fonts use
GEOM = dict(cx=320.0, pitch=89.0, ymid=86.0, slant=-0.22, sx=1.0, sy=1.0)

def seg_darkness(R, geom=GEOM):
    """darkness of every segment patch, shape (4 digits [thousands..units], 7 segments a..g)."""
    bg = np.percentile(R[:, :NW - 10], 85)
    out = np.zeros((4, 7))
    for di in range(4):
        cx = geom["cx"] - (3 - di) * geom["pitch"]
        for si, (dx, dy, kind) in enumerate(SEG.values()):
            y = geom["ymid"] + dy * geom["sy"]; x = cx + dx * geom["sx"] + geom["slant"] * (y - geom["ymid"])
            hw, hh = (7, 2) if kind == "h" else (2, 7)
            x0, x1, y0, y1 = int(round(x - hw)), int(round(x + hw)) + 1, int(round(y - hh)), int(round(y + hh)) + 1
            if x0 < 0 or y0 < 0 or y1 > R.shape[0] or x1 > R.shape[1]: out[di, si] = np.nan; continue
            out[di, si] = (bg - R[y0:y1, x0:x1].mean()) / bg
    return out

SEGS_OF = {0: "abcdef", 1: "bc", 2: "abdeg", 3: "abcdg", 4: "bcfg", 5: "acdfg", 6: "acdefg", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
GAIN = np.ones((4, 7))      # per (digit cell, segment) contrast of a LIT segment; set by calibrate() / loaded from out/lcd_gain.npy

def calibrate(Ds, values):
    """learn GAIN from frames with known readings: median darkness of each lit (cell, segment), relative to the frame's lit median."""
    acc = [[[] for _ in range(7)] for _ in range(4)]
    for D, val in zip(Ds, values):
        digs = [None] * 4; sv = str(int(val))
        for k, ch in enumerate(reversed(sv)): digs[3 - k] = int(ch)
        lit = [(c, "abcdefg".index(sg)) for c in range(4) if digs[c] is not None for sg in SEGS_OF[digs[c]]]
        if not lit or np.isnan(D).any(): continue
        ref = np.median([D[c, j] for c, j in lit])
        for c, j in lit: acc[c][j].append(D[c, j] / ref)
    G = np.array([[np.median(a) if len(a) >= 3 else np.nan for a in row] for row in acc])
    G[np.isnan(G)] = np.nanmedian(G)
    return G

def decode(D, gain=None, on=0.5, margin=0.12):
    """D from seg_darkness. Normalize by per-segment gain, scale by the frame's lit level, threshold at `on`.
    returns (value or None, confident) -- confident if no segment is within `margin` of the threshold."""
    gain = GAIN if gain is None else gain
    if np.isnan(D[1:]).any() or np.isnan(D[0, 1:3]).any(): return None, False
    Dn = D / gain
    use = np.ones((4, 7), bool); use[0, [0, 3, 4, 5, 6]] = False           # thousands: only b, c (can be blank or 1)
    vals = Dn[use]; L = np.sort(vals)[-3:].mean()                           # frame lit level = mean of the 3 darkest segments
    if L < 0.04: return None, False                                         # nothing lit at all (display off / blank)
    Z = Dn / L
    digits = []
    th = "".join("bc" if (Z[0, 1] > on and Z[0, 2] > on) else "")
    if (Z[0, 1] > on) != (Z[0, 2] > on): return None, False
    digits.append(1 if th else None)
    for c in range(1, 4):
        segs = "".join(n for n, z in zip("abcdefg", Z[c]) if z > on)
        if segs not in DIG: return None, False
        digits.append(DIG[segs])
    shown = [d is not None for d in digits]
    if not any(shown) or any(shown[i] and not shown[i + 1] for i in range(3)): return None, False
    confident = bool(np.all(np.abs(Z[use] - on) > margin))
    return int("".join(str(d) for d in digits if d is not None)), confident

def read(crop_gray, geom=GEOM, thr=0.12):
    R = register(crop_gray)
    if R is None: return None, False, None
    D = seg_darkness(R.astype(np.float32), geom); v, c = decode(D)
    return v, c, D


def read_stack(stack, gain=None):
    """decode every crop in a (n, h, w) stack -> arrays raw value (-1 = unreadable) and confident flag."""
    raw = np.full(len(stack), -1, int); conf = np.zeros(len(stack), bool)
    for k, c in enumerate(stack):
        R = register(c)
        if R is None: continue
        v, cf = decode(seg_darkness(R.astype(np.float32)), gain)
        if v is not None: raw[k] = v; conf[k] = cf
    return raw, conf

def consensus(raw, min_run=4, max_gap=3):
    """keep a value only where it is read identically in >= min_run frames in a row (unreadable frames of up to max_gap
    inside a run are tolerated). The display holds each value for several frames, so refresh transitions and one-off
    misreads drop out. returns per-frame stable value (nan where none) and the list of runs (start, end, value)."""
    runs = []; k = 0; n = len(raw)
    while k < n:
        if raw[k] < 0: k += 1; continue
        v = raw[k]; j = k; last = k; gap = 0; cnt = 0
        while j < n and gap <= max_gap:
            if raw[j] == v: last = j; cnt += 1; gap = 0
            elif raw[j] < 0: gap += 1
            else: break
            j += 1
        if cnt >= min_run: runs.append((k, last, int(v)))
        k = last + 1
    out = np.full(n, np.nan)
    for a, b, v in runs: out[a:b + 1] = v
    return out, runs
