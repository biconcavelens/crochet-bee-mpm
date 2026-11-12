#!/usr/bin/env bash
# upload the code + small data Stage B needs to /content/code on the Colab session (run inside WSL)
cd "/mnt/c/projects/dirac assignment"
echo 'import os; [os.makedirs(f"/content/code/{d}", exist_ok=True) for d in ("src", "out", "colab")]' | colab exec -s beesplat > /dev/null 2>&1
for f in src/mpm_bee.py src/fit_dense.py src/fit_mpm_dense.py out/series.csv out/fit3d_dense_best.json \
         colab/stageB_align.py colab/stageB_orbit.py colab/stageB_physgauss.py; do
  colab upload -s beesplat "$f" "/content/code/$f" 2>&1 | grep -a "Uploaded\|rror" | tail -1
done
