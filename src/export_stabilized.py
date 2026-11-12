"""Export the stabilized input clip (every frame warped onto the reference view, as used for all measurements), 30 fps.
usage: python src/export_stabilized.py  -> media/input_stepped_presses_stabilized.mp4   (needs the raw 60 fps clip)
"""
import sys, glob, os, subprocess, cv2
sys.path.insert(0, "src")
from stabilize import VIDEO, REF_T, feats, homography, frame_at

cap = cv2.VideoCapture(VIDEO); ref = feats(frame_at(cap, REF_T)); cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
W, H = 1080, 1920; raw = "out/_raw_stab.mp4"; Hm = None; k = 0
vid = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
while cap.grab():
    if k % 2 == 0:                                   # 60 -> 30 fps
        ok, fr = cap.retrieve(); h, n = homography(ref, feats(fr))
        if h is not None and n >= 60: Hm = h         # keep the last good warp if a frame has too few static features
        if Hm is not None: vid.write(cv2.warpPerspective(fr, Hm, (W, H)))
        if k % 1200 == 0: print(f"  {k / 60:.0f} s", flush=True)
    k += 1
vid.release()
FFMPEG = (glob.glob(os.path.expanduser("~/AppData/Local/Programs/Python/*/Lib/site-packages/imageio_ffmpeg/binaries/ffmpeg*.exe")) + ["ffmpeg"])[0]
subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", raw, "-c:v", "libx264", "-crf", "28", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "media/input_stepped_presses_stabilized.mp4"], check=True)
os.remove(raw); print("done")
