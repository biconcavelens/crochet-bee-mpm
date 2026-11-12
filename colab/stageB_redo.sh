#!/usr/bin/env bash
# fresh L4 session -> stage B env -> upload code + aligned splat -> render (arg 1 = seconds of the clip, default all)
# run inside WSL:  bash colab/stageB_redo.sh [seconds]
set -e
cd "/mnt/c/projects/dirac assignment"
export COLAB_CLI_REQUEST_TIMEOUT=900 COLAB_CLI_ASSIGN_RETRIES=0
q() { grep -a -v "update\|silence\|uv tool" || true; }
if ! colab sessions 2>&1 | grep -q "^\[beesplat\]"; then colab new -s beesplat --gpu L4 2>&1 | q | tail -n 2; fi
colab exec -s beesplat -f colab/stageB_setup.py 2>&1 | q | grep -a -E "ok |DONE|FAILED|rror" || true
echo 'import os; [os.makedirs(f"/content/code/{d}", exist_ok=True) for d in ("src", "out", "colab")]' | colab exec -s beesplat > /dev/null 2>&1
for f in src/mpm_bee.py src/fit_dense.py src/fit_mpm_dense.py out/series.csv out/frames.csv out/fit3d_dense_best.json colab/stageB_physgauss.py; do
  colab upload -s beesplat "$f" "/content/code/$f" 2>&1 | grep -a -c Uploaded > /dev/null
done
colab upload -s beesplat splat/bee_sim.npz /content/bee_sim.npz 2>&1 | grep -a Uploaded
echo '{}' > /tmp/presscam.json; colab upload -s beesplat /tmp/presscam.json /content/presscam.json > /dev/null 2>&1
echo "cd /content/code && rm -f /content/physgauss.log && nohup /content/ns/bin/python colab/stageB_physgauss.py ${1:-} > /content/physgauss.log 2>&1 &" | timeout 60 colab console -s beesplat > /dev/null 2>&1
bash colab/wait_done.sh /content/physgauss.log "PHYSGAUSS DONE" 60 || { echo "tail -n 20 /content/physgauss.log" | timeout 60 colab console -s beesplat 2>&1 | tr -d "\r" | tail -n 20; exit 1; }
bash colab/download.sh splat /content/photoreal_sim.mp4
