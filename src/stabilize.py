"""Warp each frame of the handheld press clip onto a reference frame, using only static features (scale body + platform disk).
usage: python src/stabilize.py  -> out/steps_stab_H.npy (per-sample 3x3 homographies to reference) + check mosaic
Features come from y in [Y0,Y1] (full-res) so bee/pusher/hand/background are excluded; RANSAC rejects the rest.
"""
import cv2, numpy as np
VIDEO = "presses/5 second long presses.mp4"
REF_T = 75.0            # a 0 g frame with the whole disk visible
Y0, Y1, SC = 760, 1700, 0.5
sift = cv2.SIFT_create(4000)

def feats(img):
    g = cv2.cvtColor(cv2.resize(img, None, fx=SC, fy=SC), cv2.COLOR_BGR2GRAY)
    m = np.zeros_like(g); m[int(Y0*SC):int(Y1*SC)] = 255
    k, d = sift.detectAndCompute(g, m)
    return np.float32([p.pt for p in k]) / SC, d

def homography(ref, cur):
    (kp0, d0), (kp1, d1) = ref, cur
    ms = cv2.BFMatcher().knnMatch(d1, d0, k=2)
    good = [a for a, b in (m for m in ms if len(m) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 12: return None, 0
    src = kp1[[g.queryIdx for g in good]]; dst = kp0[[g.trainIdx for g in good]]
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    return H, int(inl.sum()) if inl is not None else 0

def frame_at(cap, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * cap.get(cv2.CAP_PROP_FPS)))); ok, f = cap.read(); return f if ok else None

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO); ref_img = frame_at(cap, REF_T); ref = feats(ref_img)
    ts = [float(x) for x in np.genfromtxt("out/steps_force_reads.csv", delimiter=",", skip_header=1, usecols=0)]
    tiles, res = [], {}
    for t in ts:
        f = frame_at(cap, t); H, n = homography(ref, feats(f)); res[t] = (H, n)
        print(f"{t:5.0f}s inliers={n}")
        if H is not None:
            w = cv2.warpPerspective(f, H, (1080, 1920)); tiles.append(cv2.resize(w[300:1300], (270, 250)))
    np.save("out/steps_stab_H.npy", np.array([(t, *(res[t][0].ravel() if res[t][0] is not None else np.zeros(9)), res[t][1]) for t in ts]))
    while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite("out/steps_stab_check.png", np.vstack([np.hstack(tiles[i:i+8]) for i in range(0, len(tiles), 8)]))
