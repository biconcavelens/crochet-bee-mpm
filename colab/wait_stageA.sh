#!/usr/bin/env bash
# poll the detached Stage A job until it finishes or fails (run inside WSL)
cd "/mnt/c/projects/dirac assignment"
last=""
for i in $(seq 1 60); do
  s=$(colab exec -s beesplat -f colab/stageA_status.py 2>&1 | grep -v "update\|silence\|uv tool")
  steps=$(echo "$s" | grep -E "^STEP|registered|ALL DONE" | tr '\n' ' ')
  [ "$steps" != "$last" ] && echo "[$(date +%H:%M)] $steps" && last="$steps"
  if echo "$s" | grep -q "ALL DONE"; then echo "$s"; exit 0; fi
  if echo "$s" | grep -qiE "Traceback|Error:|FAILED|Killed"; then echo "PROBLEM:"; echo "$s"; exit 1; fi
  sleep 120
done
echo "timed out waiting"; echo "$s"
