"""Stage A setup on the Colab VM (run with: colab exec -s beesplat -f colab/stageA_setup.py).
The Colab image is Python 3.13 + torch built for CUDA 13; nerfstudio needs Python <= 3.11 and gsplat's prebuilt kernels need a
CUDA 12 torch, so a separate uv environment is created at /content/ns: Python 3.10 (gsplat 1.4.0 wheels are cp310 only), torch 2.4.1+cu124, gsplat 1.4.0 (prebuilt
wheel), nerfstudio. COLMAP comes from apt (3.9.1, CPU feature extraction/matching is fine for ~300 images)."""
import subprocess, sys

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    out = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join(out[-15:]), flush=True)
    if r.returncode != 0: raise SystemExit(f"FAILED ({r.returncode}): {cmd}")

sh("apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y colmap ffmpeg > /dev/null && colmap -h | head -2")
sh("rm -rf /content/ns && uv venv /content/ns --python 3.10 -q")
PY = "/content/ns/bin/python"
sh(f"uv pip install -q --python {PY} torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124")
sh(f"uv pip install -q --python {PY} 'gsplat==1.4.0+pt24cu124' --extra-index-url https://docs.gsplat.studio/whl/pt24cu124 --index-strategy unsafe-best-match")
sh(f"uv pip install -q --python {PY} nerfstudio==1.1.5")
sh(f"{PY} -c \"import torch, gsplat, nerfstudio; print('torch', torch.__version__, 'cuda ok', torch.cuda.is_available(), torch.cuda.get_device_name(0)); print('gsplat', gsplat.__version__)\"")
# smoke test: gsplat's CUDA rasterizer actually runs on this GPU
sh(f"""{PY} -c "
import torch, gsplat
N = 1000; d = 'cuda'
m = torch.randn(N, 3, device=d) * 0.3 + torch.tensor([0, 0, 3.0], device=d)
q = torch.nn.functional.normalize(torch.randn(N, 4, device=d), dim=-1); s = torch.full((N, 3), 0.02, device=d)
o = torch.ones(N, device=d) * 0.8; c = torch.rand(N, 3, device=d)
K = torch.tensor([[300., 0, 160], [0, 300., 120], [0, 0, 1]], device=d)[None]; V = torch.eye(4, device=d)[None]
img, a, _ = gsplat.rasterization(m, q, s, o, c, V, K, 320, 240)
print('rasterization ok', tuple(img.shape), float(img.mean()))
" """)
sh("ls /content/ns/bin | grep -E '^ns-(process-data|train|export|render)$'")
print("\nSETUP DONE")
