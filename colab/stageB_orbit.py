"""Orbit preview of the aligned splat (sim frame, mm): 360 deg around the bee, 20 deg above horizontal, 1280x720, 6 s.
in: /content/bee_sim.npz   out: /content/splat_orbit.mp4
run: /content/ns/bin/python /content/code/colab/stageB_orbit.py"""
import subprocess, numpy as np, torch, gsplat, cv2
dev = "cuda"; Z = np.load("/content/bee_sim.npz")
def T(a): return torch.tensor(a, dtype=torch.float32, device=dev)
m, q, s, o = T(Z["means"]), T(Z["quats"]), torch.exp(T(Z["log_scales"])), torch.sigmoid(T(Z["opacity"]))
c = torch.cat([T(Z["sh_dc"])[:, None], T(Z["sh_rest"])], 1); deg = int(np.sqrt(c.shape[1])) - 1
ZC, FLOOR = 64.0, 6.0; tgt = np.array([ZC, FLOOR + 20, ZC]); W, H, f = 1280, 720, 1300.0
vid = cv2.VideoWriter("/content/_orbit.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
for k in range(180):
    a = 2 * np.pi * k / 180; el = np.radians(20); d = 260.0
    pos = tgt + d * np.array([np.sin(a) * np.cos(el), np.sin(el), -np.cos(a) * np.cos(el)])   # starts in front of the face
    fwd = (tgt - pos) / d; right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); up = np.cross(right, fwd)
    w2c = np.eye(4); w2c[:3, :3] = np.stack([right, -up, fwd]); w2c[:3, 3] = -w2c[:3, :3] @ pos             # OpenCV camera
    img, _, _ = gsplat.rasterization(m, q, s, o, c, T(w2c)[None], T([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])[None], W, H,
                                     sh_degree=deg, backgrounds=T([[1.0, 1.0, 1.0]]))
    vid.write((img[0].clamp(0, 1).cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8))
vid.release()
subprocess.run("ffmpeg -y -loglevel error -i /content/_orbit.mp4 -c:v libx264 -pix_fmt yuv420p -crf 18 /content/splat_orbit.mp4", shell=True, check=True)
print("ORBIT DONE")
