#!/usr/bin/env bash
# upload the orbit clips to the Colab session (run inside WSL: bash colab/upload_orbit.sh)
set -e
SESSION=${1:-beesplat}
cd "/mnt/c/projects/dirac assignment/orbit"
for f in *.mp4; do
  echo "uploading $f"
  colab upload -s "$SESSION" "$f" "/content/orbit/$f" 2>&1 | grep -v "update\|silence\|uv tool" | tail -1
done
echo 'import glob, os; print([(os.path.basename(p), os.path.getsize(p)) for p in sorted(glob.glob("/content/orbit/*.mp4"))])' | colab exec -s "$SESSION" 2>&1 | grep -v "update\|silence\|uv tool"
