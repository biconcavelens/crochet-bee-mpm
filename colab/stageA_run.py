"""Stage A on the Colab VM, launched detached (survives CLI disconnects): frames -> blur filter -> COLMAP -> splatfacto ->
export. Progress goes to /content/stageA.log; each step writes a marker line 'STEP <name> DONE'.
launch:  colab exec -s beesplat -f colab/stageA_run.py      (returns immediately)
status:  colab exec -s beesplat -f colab/stageA_status.py
"""
import subprocess, textwrap

JOB = textwrap.dedent(r'''
set -e
exec > /content/stageA.log 2>&1
NS=/content/ns/bin
echo "STEP start $(date)"

# 1. frames: 4 per second from every clip (~260), drop the blurriest 15% (variance of the Laplacian)
rm -rf /content/frames && mkdir -p /content/frames
k=0; for c in /content/orbit/*.mp4; do ffmpeg -loglevel error -i "$c" -vf fps=4 -q:v 2 /content/frames/clip${k}_%04d.jpg; k=$((k+1)); done
$NS/python - <<'PY'
import cv2, glob, os, numpy as np
imgs = sorted(glob.glob('/content/frames/*.jpg'))
s = {p: cv2.Laplacian(cv2.imread(p, 0), cv2.CV_64F).var() for p in imgs}
cut = np.percentile(list(s.values()), 15)
for p, v in s.items():
    if v < cut: os.remove(p)
print(len(imgs), 'frames,', len(glob.glob('/content/frames/*.jpg')), 'kept')
PY
echo "STEP frames DONE"

# 2. camera poses (COLMAP via nerfstudio). vocab-tree matching scales across clips; fall back to exhaustive if it fails
$NS/ns-process-data images --data /content/frames --output-dir /content/bee --matching-method vocab_tree --no-gpu --num-downscales 2 \
  || $NS/ns-process-data images --data /content/frames --output-dir /content/bee --matching-method exhaustive --no-gpu --num-downscales 2
$NS/python -c "import json,glob; T=json.load(open('/content/bee/transforms.json')); print('registered', len(T['frames']), 'of', len(glob.glob('/content/frames/*.jpg')))"
echo "STEP colmap DONE"

# 3. 3D Gaussian Splatting (splatfacto) at half resolution
$NS/ns-train splatfacto --data /content/bee --max-num-iterations 20000 \
   --pipeline.datamanager.camera-res-scale-factor 0.5 \
   --viewer.quit-on-train-completion True --vis tensorboard --output-dir /content/outputs
echo "STEP train DONE"

# 4. export the Gaussians + an orbit preview through the training cameras
CFG=$(ls -t /content/outputs/bee/splatfacto/*/config.yml | head -1)
mkdir -p /content/export
$NS/ns-export gaussian-splat --load-config $CFG --output-dir /content/export
$NS/ns-render interpolate --load-config $CFG --output-path /content/export/orbit_preview.mp4 --frame-rate 30 --interpolation-steps 3 --downscale-factor 2
cp /content/bee/transforms.json /content/export/
ls -la /content/export
echo "STEP export DONE"
echo "ALL DONE $(date)"
''')
open("/content/stageA.sh", "w").write(JOB)
subprocess.Popen("nohup bash /content/stageA.sh > /dev/null 2>&1 &", shell=True)
print("stage A launched; log: /content/stageA.log")
