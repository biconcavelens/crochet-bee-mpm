#!/usr/bin/env bash
# download files from the Colab session into the project (run inside WSL):  bash colab/download.sh <dest dir> <remote file>...
# (downloads land in WSL's own /tmp first: writing directly onto the Windows drive fails)
DEST="/mnt/c/projects/dirac assignment/$1"; shift
mkdir -p "$DEST" /tmp/colab_dl
for r in "$@"; do
  b=$(basename "$r")
  colab download -s beesplat "$r" "/tmp/colab_dl/$b" 2>&1 | grep -a "Downloaded\|rror" | tail -1
  cp "/tmp/colab_dl/$b" "$DEST/$b" && echo "  -> $1/$b ($(stat -c %s "$DEST/$b") bytes)"
done
