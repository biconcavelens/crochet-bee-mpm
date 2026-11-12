"""Dump the exact training cameras nerfstudio used (same frame as the exported splat), with their image names.
run on the VM: /content/ns/bin/python /content/code/colab/dump_cams.py  -> /content/export/train_cams.json"""
import glob, json
from pathlib import Path
from nerfstudio.utils.eval_utils import eval_setup

cfg = Path(sorted(glob.glob("/content/outputs/bee/splatfacto/*/config.yml"))[-1])
_, pipeline, _, _ = eval_setup(cfg, test_mode="inference")
out = []
for split, dpo in (("train", pipeline.datamanager.train_dataparser_outputs), ("eval", getattr(pipeline.datamanager, "eval_dataparser_outputs", None))):
    if dpo is None: continue
    cams = dpo.cameras
    for i, fn in enumerate(dpo.image_filenames):
        out.append(dict(split=split, file=Path(fn).name, c2w=cams.camera_to_worlds[i].tolist(),
                        fx=float(cams.fx[i]), fy=float(cams.fy[i]), cx=float(cams.cx[i]), cy=float(cams.cy[i]),
                        w=int(cams.width[i]), h=int(cams.height[i])))
json.dump(out, open("/content/export/train_cams.json", "w"))
print("CAMS DONE", len(out))
