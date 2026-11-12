#!/usr/bin/env bash
# wait (via the shell console, which works even while the Python kernel is busy) until a marker appears in a VM log
# usage: bash colab/wait_done.sh <log path on VM> <marker> [max minutes]
LOG=$1; MARK=$2; MAX=${3:-60}
for i in $(seq 1 "$MAX"); do
  out=$(echo "grep -a -c '$MARK' $LOG; grep -a -c -iE 'Traceback|Error' $LOG" | timeout 120 colab console -s beesplat 2>&1 | tr -d '\r' | grep -a -E '^[0-9]+$')
  done_n=$(echo "$out" | head -1); err_n=$(echo "$out" | tail -1)
  if [ "${done_n:-0}" -ge 1 ]; then echo "found '$MARK' after ~$i min"; exit 0; fi
  if [ "${err_n:-0}" -ge 1 ]; then echo "error lines in $LOG"; exit 1; fi
  sleep 60
done
echo "timed out"
