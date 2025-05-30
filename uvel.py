# train_u.py —— 自回归 + 速度特征 + 纯预测 (ptf=0)
# -*- coding: utf-8 -*-
"""
自回归时序 GNN — 单输出 Ux,Uy,Uz
输入特征：(x,y,z, U_prev_x,y,z, α, force_flag, V_prev_x,y,z)
force_flag 从 npz 读取，V_prev 第一帧取 0
使用纯自回归 (ptf=0)
"""
import os, glob, random
import numpy as np, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import GConvGRU

# ===== 0. 超参 & 路径 =====
NPZ_DIR   = r"D:\BaiduNetdiskDownload\npz"
HIDDEN    = 128
EPOCHS    = 2000
CLIP_NORM = 0.5
LR        = 1e-3
SAVE_PATH = r"C:\py\modelu_speed.pth"
SEED      = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ===== 1. 工具函数 =====
def load_npz(path):
    d = np.load(path)
    return (
        d["node_coords"].astype(np.float32),
        d["connectivity"].astype(np.int32),
        d["frame_times"].astype(np.float32),
        d["disp"].astype(np.float32),
        d["node_labels"].astype(np.int32),
        d.get("SURF1_NODE_LABELS", None)
    )

def build_edge_index(conn, labels):
    idx = {lbl: i for i, lbl in enumerate(labels)}
    edges = set()
    for elem in conn:
        if len(elem) == 8:
            b, t = elem[:4], elem[4:]
            for k in range(4):
                edges |= {(idx[b[k]], idx[b[(k+1)%4]]),
                          (idx[t[k]], idx[t[(k+1)%4]]),
                          (idx[b[k]], idx[t[k]])}
        else:
            m = len(elem)
            for k in range(m):
                i0, i1 = elem[k], elem[(k+1)%m]
                edges.add((idx[i0], idx[i1]))
    edges |= {(j, i) for i, j in edges}
    return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

# ===== 2. 读取 npz 并组装 graphs =====
graphs = []
for fn in sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz"))):
    coord, conn, times, disp, labels, surf_nodes = load_npz(fn)
    N, T = coord.shape[0], disp.shape[0]
    # 计算速度序列
    dt = np.diff(times, prepend=times[0])               # (T,)
    vel = np.zeros_like(disp)                           # (T,N,3)
    vel[1:] = (disp[1:] - disp[:-1]) / dt[1:,None,None]
    # 构造边
    edge = build_edge_index(conn, labels).to(device)
    # force flag
    if surf_nodes is None:
        flag = np.zeros(N, dtype=np.float32)
    else:
        surf_set = set(int(x) for x in surf_nodes)
        flag = np.array([1.0 if int(lbl) in surf_set else 0.0 for lbl in labels], dtype=np.float32)
    flag_t = torch.tensor(flag, device=device).unsqueeze(1)  # (N,1)
    # 构造 X: (T,N,11)
    X = torch.zeros((T, N, 11), dtype=torch.float32, device=device)
    coord_t = torch.tensor(coord, device=device)
    for t in range(T):
        alpha_col = torch.full((N,1), times[t], device=device)
        u_prev = torch.zeros((N,3), device=device) if t==0 else torch.tensor(disp[t-1], device=device)
        v_prev = torch.zeros((N,3), device=device) if t==0 else torch.tensor(vel[t-1], device=device)
        X[t] = torch.cat([coord_t, u_prev, alpha_col, flag_t, v_prev], dim=1)
    Y_u = torch.tensor(disp, device=device)
    # 保存 frame_times 以供打印
    graphs.append({"X":X, "edge":edge, "Y_u":Y_u, "times":times})
print(f"Loaded {len(graphs)} npz files.")

# ===== 3. 计算 μ/σ（标准化） =====
feat_all = torch.cat([g["X"].reshape(-1,11).cpu() for g in graphs], dim=0)
out_all  = torch.cat([g["Y_u"].reshape(-1,3).cpu() for g in graphs], dim=0)
mu_x, std_x = feat_all.mean(0), feat_all.std(0)
mu_u, std_u = out_all.mean(0), out_all.std(0)
# alpha, flag, v_prev 列不归一化：索引 6,7,8,9,10?
# 实际上只要保留索引 6(alpha)与7(flag)不归一化，速度可归一化
mu_x[6], std_x[6] = 0.0, 1.0
mu_x[7], std_x[7] = 0.0, 1.0
std_x[std_x==0] = 1.0; std_u[std_u==0] = 1.0
for g in graphs:
    g["X"]   = (g["X"]   - mu_x.to(device)) / std_x.to(device)
    g["Y_u"] = (g["Y_u"] - mu_u.to(device)) / std_u.to(device)

# ===== 4. 定义模型 =====
class AutoregU(nn.Module):
    def __init__(self, in_f=11, h=HIDDEN, out_f=3, K=1):
        super().__init__()
        self.gru  = GConvGRU(in_f, h, K=K)
        self.head = nn.Linear(h, out_f)
    def forward(self, X_seq, edge, Y_u=None, p_tf=0.0):
        X, h, outs = [x.clone() for x in X_seq], None, []
        for t in range(len(X)):
            h = self.gru(X[t], edge_index=edge, H=h)
            u = self.head(h)
            outs.append(u)
            # 纯自回归 p_tf=0
            if t < len(X)-1:
                X[t+1][:,3:6]  = u.detach()                   # U_prev
                # 更新速度预测: v = (u - u_prev) / dt
                # 用 dt from times
                dt = X_seq[t+1][:,6] - X_seq[t][:,6]
                v = (u.detach() - X_seq[t][:,3:6]) / dt.unsqueeze(1)
                X[t+1][:,8:11] = v
        return torch.stack(outs)
model = AutoregU().to(device)

# ===== 5. 优化器 =====
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

# ===== 6. 训练 =====
hist=[]
for epoch in range(1, EPOCHS+1):
    epoch_loss=0.0
    for g in graphs:
        out = model(g["X"], g["edge"], g["Y_u"], p_tf=0.0)
        loss = F.mse_loss(out, g["Y_u"])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        opt.step(); epoch_loss+=loss.item()
    hist.append(epoch_loss)
    if epoch==1 or epoch%10==0:
        print(f"Ep{epoch:04d} loss={epoch_loss:.4f}")

# ===== 7. 每帧平均 Loss 打印 =====
print("\nPer-frame average MSE:")
f_times = graphs[0]["times"]
pf = np.zeros(len(f_times), dtype=np.float32)
model.eval()
with torch.no_grad():
    for g in graphs:
        out = model(g["X"], g["edge"], None, p_tf=0.0)
        pf += (out - g["Y_u"]).pow(2).mean(dim=(1,2)).cpu().numpy()
    pf /= len(graphs)
for t,l in enumerate(pf): print(f"Frame {t:3d} Time={f_times[t]:.3f} Loss={l:.6f}")

# ===== 8. 保存模型 =====
torch.save({"model":model.state_dict(),
            "mu_x":mu_x, "std_x":std_x,
            "mu_u":mu_u, "std_u":std_u}, SAVE_PATH)
print("Model saved to", SAVE_PATH)
