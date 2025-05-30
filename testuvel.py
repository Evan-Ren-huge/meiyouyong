#!/usr/bin/env python3
# predict_u_speed.py —— 预测加入速度特征的 Ux,Uy,Uz
# --------------------------------------------
# 用法：修改下面三条路径后直接运行

import os, numpy as np, torch
from torch_geometric_temporal.nn.recurrent import GConvGRU
import torch.nn as nn

# ===== 0. 路径设置 =====
NPZ_PATH = r"D:\BaiduNetdiskDownload\npz\Job-t2.npz"    # 真值 npz
MODEL_U  = r"C:\py\modelu_speed.pth"                       # 训练好的模型
OUT_NPZ  = NPZ_PATH.replace('.npz', '_pred_speed.npz')        # 输出路径

# ===== 1. 初始化 =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

# ===== 2. 辅助：构造边 =====
def build_edge_index(conn, labels):
    idx = {lbl: i for i, lbl in enumerate(labels)}
    edges = set()
    for elem in conn:
        if len(elem) == 8:
            b, t = elem[:4], elem[4:]
            for k in range(4):
                edges |= {
                    (idx[b[k]], idx[b[(k+1)%4]]),
                    (idx[t[k]], idx[t[(k+1)%4]]),
                    (idx[b[k]], idx[t[k]])
                }
        else:
            m = len(elem)
            for k in range(m):
                i0, i1 = elem[k], elem[(k+1)%m]
                edges.add((idx[i0], idx[i1]))
    edges |= {(j, i) for i, j in edges}
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

# ===== 3. 载入数据 =====
d = np.load(NPZ_PATH)
coord       = d["node_coords"].astype(np.float32)
conn        = d["connectivity"].astype(np.int32)
labels      = d["node_labels"].astype(np.int32)
times       = d["frame_times"].astype(np.float32)
disp        = d["disp"].astype(np.float32)
surf_nodes  = d.get("SURF1_NODE_LABELS", None)

# 归一化 times 不必做——直接使用实际帧时间差
T, N = len(times), coord.shape[0]
edge = build_edge_index(conn, labels).to(device)

# ===== 4. 计算速度（基于真值） =====
dt  = np.diff(times, prepend=times[0])       # (T,)
vel = np.zeros_like(disp)                   # (T,N,3)
vel[1:] = (disp[1:] - disp[:-1]) / dt[1:, None, None]

# ===== 5. 构造 force_flag =====n
if surf_nodes is None:
    flag = np.zeros(N, dtype=np.float32)
else:
    surf_set = set(int(x) for x in surf_nodes)
    flag = np.array([1.0 if int(lbl) in surf_set else 0.0 for lbl in labels], dtype=np.float32)
flag_t = torch.tensor(flag, device=device).unsqueeze(1)  # (N,1)

# ===== 6. 构造输入序列 X: (T,N,11) =====
X = torch.zeros((T, N, 11), dtype=torch.float32, device=device)
coord_t = torch.tensor(coord, device=device)
for t in range(T):
    alpha_col = torch.full((N,1), times[t], device=device)
    if t == 0:
        u_prev = torch.zeros((N,3), device=device)
        v_prev = torch.zeros((N,3), device=device)
    else:
        u_prev = torch.tensor(disp[t-1], device=device)
        v_prev = torch.tensor(vel[t-1], device=device)
    X[t] = torch.cat([coord_t, u_prev, alpha_col, flag_t, v_prev], dim=1)

# ===== 7. 载入模型 =====
ckpt = torch.load(MODEL_U, map_location=device)
class AutoregU(nn.Module):
    def __init__(self, in_f=11, h=128, out_f=3, K=1):
        super().__init__()
        self.gru  = GConvGRU(in_f, h, K=K)
        self.head = nn.Linear(h, out_f)
    def forward(self, X_seq, edge, Y_u=None, p_tf=0.0):
        X, h, outs = [x.clone() for x in X_seq], None, []
        for t in range(len(X)):
            h = self.gru(X[t], edge_index=edge, H=h)
            u = self.head(h)
            outs.append(u)
            if t < len(X)-1:
                X[t+1][:,3:6]  = u.detach()
                dt_t = X_seq[t+1][:,6] - X_seq[t][:,6]
                v = (u.detach() - X_seq[t][:,3:6]) / dt_t.unsqueeze(1)
                X[t+1][:,8:11] = v
        return torch.stack(outs)
model = AutoregU().to(device)
model.load_state_dict(ckpt["model"])
mu_x, std_x = ckpt["mu_x"].to(device), ckpt["std_x"].to(device)
mu_u, std_u = ckpt["mu_u"].to(device), ckpt["std_u"].to(device)

# ===== 8. 归一化输入 =====
X_n = (X - mu_x) / std_x

# ===== 9. 预测 =====
model.eval()
with torch.no_grad():
    pred_u_n = model(X_n, edge, None, p_tf=0.0)  # (T,N,3)
pred_u   = pred_u_n * std_u + mu_u             # 反归一化

print(f"✓ U 预测完成，shape = {pred_u.shape}")

# ===== 10. 保存结果 =====
src = dict(d)
src["pred_u"] = pred_u.cpu().numpy()
np.savez_compressed(OUT_NPZ, **src)
print("✅ 保存到:", OUT_NPZ)
