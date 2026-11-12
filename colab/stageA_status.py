"""Print Stage A progress from the VM log (markers + the last few lines, with carriage-return progress bars collapsed)."""
import os, re
p = "/content/stageA.log"
if not os.path.exists(p): print("no log yet")
else:
    txt = open(p, errors="ignore").read()
    lines = [l.split("\r")[-1] for l in txt.splitlines() if l.strip()]
    print("\n".join(l for l in lines if l.startswith(("STEP", "ALL DONE", "registered")) or "kept" in l))
    print("--- tail ---"); print("\n".join(re.sub(r"\x1b\[[0-9;]*m", "", l)[:160] for l in lines[-8:]))
