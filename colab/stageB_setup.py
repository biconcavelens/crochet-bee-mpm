"""Stage B environment on a fresh Colab VM (run with: colab exec -s beesplat -f colab/stageB_setup.py).
Only what the PhysGaussian render needs: Python 3.10 venv with torch 2.4.1+cu124, gsplat 1.4.0 (prebuilt cp310 wheel),
taichi 1.7.4, scipy, opencv; plus ffmpeg."""
import subprocess

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-8:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"FAILED ({r.returncode}): {cmd}")

PY = "/content/ns/bin/python"
sh("apt-get -qq update > /dev/null; DEBIAN_FRONTEND=noninteractive apt-get -qq install -y ffmpeg > /dev/null; ffmpeg -version | head -1")
sh("rm -rf /content/ns && uv venv /content/ns --python 3.10 -q")
sh(f"uv pip install -q --python {PY} torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124")
sh(f"uv pip install -q --python {PY} 'gsplat==1.4.0+pt24cu124' --extra-index-url https://docs.gsplat.studio/whl/pt24cu124 --index-strategy unsafe-best-match")
sh(f"uv pip install -q --python {PY} taichi==1.7.4 scipy opencv-python-headless setuptools 'numpy<2'")
sh(f"{PY} -c \"import torch, gsplat, taichi, scipy, cv2; print('ok', torch.__version__, torch.cuda.get_device_name(0), gsplat.__version__, taichi.__version__)\"")
print("STAGE B SETUP DONE")
