# train_u_ps.py —— 双 GRU：U 与 (S+PEEQ)
# --------------------------------------------------
# 输入特征: 11 维 (x,y,z, U_prev, α, flag, V_prev)
# 任务 1: 预测 Ux,Uy,Uz          (3 维)
# 任务 2: 预测 S_vm (=von Mises) (1 维)
# 任务 3: 预测 PEEQ              (1 维)
# --------------------------------------------------
import os, glob, random, numpy as np, torch
import torch.nn as nn, torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric_temporal.nn.recurrent import GConvGRU

# ===== 0. 路径与超参 =====
NPZ_DIR = r"/root/autodl-tmp"      # 改成你的路径
SAVE_PATH = r"/root/autodl-tmp/model_u_ps.pth"
HIDDEN    = 256
EPOCHS    = 2000
LR        = 1e-3
CLIP_NORM = 0.5
W_S, W_P  = 3, 5                   # S、PEEQ 损失权重
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
        d["disp"].astype(np.float32),      # U
        d["s"].astype(np.float32),         # von Mises
        d["peeq"].astype(np.float32),      # PEEQ
        d["node_labels"].astype(np.int32),
        d.get("SURF1_NODE_LABELS", None)
    )

def build_edge_index(conn, labels):
    idx = {lbl:i for i,lbl in enumerate(labels)}
    edges=set()
    for e in conn:
        if len(e)==8:
            b,t=e[:4],e[4:]
            for k in range(4):
                edges|={(idx[b[k]],idx[b[(k+1)%4]]),
                        (idx[t[k]],idx[t[(k+1)%4]]),
                        (idx[b[k]],idx[t[k]])}
        else:
            m=len(e)
            for k in range(m):
                edges.add((idx[e[k]],idx[e[(k+1)%m]]))
    edges|={(j,i) for i,j in edges}
    return torch.tensor(list(edges),dtype=torch.long).t().contiguous()

# ===== 2. 读取 npz 并组装 =====
graphs=[]
for fn in sorted(glob.glob(os.path.join(NPZ_DIR,"*.npz"))):
    coord,conn,times,disp,s,peeq,labels,surf=load_npz(fn)
    N,T=coord.shape[0],disp.shape[0]
    # 速度
    dt=np.diff(times,prepend=times[0])
    vel=np.zeros_like(disp); vel[1:]=(disp[1:]-disp[:-1])/dt[1:,None,None]
    edge=build_edge_index(conn,labels).to(device)
    # flag
    flag=np.zeros(N,dtype=np.float32)
    if surf is not None:
        surf=set(int(x) for x in surf)
        flag=np.array([1.0 if int(lbl) in surf else 0.0 for lbl in labels],dtype=np.float32)
    flag_t=torch.tensor(flag,device=device).unsqueeze(1)
    # 构造 X
    X=torch.zeros((T,N,11),dtype=torch.float32,device=device)
    coord_t=torch.tensor(coord,device=device)
    for t in range(T):
        alpha=torch.full((N,1),times[t],device=device)
        if t==0:
            u_prev=v_prev=torch.zeros((N,3),device=device)
        else:
            u_prev=torch.tensor(disp[t-1],device=device)
            v_prev=torch.tensor(vel[t-1],device=device)
        X[t]=torch.cat([coord_t,u_prev,alpha,flag_t,v_prev],dim=1)
    graphs.append({
        "X":X,
        "edge":edge,
        "Y_u":torch.tensor(disp,device=device),
        "Y_s":torch.tensor(s,device=device),
        "Y_p":torch.tensor(peeq,device=device)
    })
print("Loaded",len(graphs),"npz files")

# ===== 3. 归一化 =====
feat_all=torch.cat([g["X"].reshape(-1,11).cpu() for g in graphs],0)
u_all  =torch.cat([g["Y_u"].reshape(-1,3).cpu() for g in graphs],0)
s_all  =torch.cat([g["Y_s"].reshape(-1   ).cpu() for g in graphs],0)
p_all  =torch.cat([g["Y_p"].reshape(-1   ).cpu() for g in graphs],0)

mu_x,std_x=feat_all.mean(0),feat_all.std(0)
mu_u,std_u=u_all.mean(0),u_all.std(0)
mu_s,std_s=s_all.mean(), s_all.std()
mu_p,std_p=p_all.mean(), p_all.std()

# 不归一化 alpha(6) 与 flag(7)
mu_x[6]=mu_x[7]=0.0; std_x[6]=std_x[7]=1.0
std_x[std_x==0]=1.0; std_u[std_u==0]=1.0
std_s=std_s or 1.0; std_p=std_p or 1.0

for g in graphs:
    g["X"]   = (g["X"]-mu_x.to(device))/std_x.to(device)
    g["Y_u"] = (g["Y_u"]-mu_u.to(device))/std_u.to(device)
    g["Y_s"] = (g["Y_s"]-mu_s)/std_s
    g["Y_p"] = (g["Y_p"]-mu_p)/std_p

# ===== 4. 模型：两条 GRU 分支 =====
class MultiGRU(nn.Module):
    def __init__(self,in_f=11,h=HIDDEN):
        super().__init__()
        self.backbone = GConvGRU(in_f,h,K=1)  # 共享一层
        self.gru_u    = GConvGRU(h,h,K=1)
        self.gru_sp   = GConvGRU(h,h,K=1)     # S & PEEQ 共用
        self.head_u = nn.Linear(h,3)
        self.head_s = nn.Linear(h,1)
        self.head_p = nn.Linear(h,1)
    def forward(self,X_seq,edge):
        h0=None
        outs_u,outs_s,outs_p=[],[],[]
        for x in X_seq:
            h0=self.backbone(x,edge_index=edge,H=h0)
            hu=self.gru_u(h0,edge_index=edge)
            hp=self.gru_sp(h0,edge_index=edge)
            outs_u.append(self.head_u(hu))           # (N,3)
            outs_s.append(self.head_s(hp).squeeze(-1))  # (N,)
            outs_p.append(self.head_p(hp).squeeze(-1))  # (N,)
        return (torch.stack(outs_u),
                torch.stack(outs_s),
                torch.stack(outs_p))

model=MultiGRU().to(device)
opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)

# ===== 5. 训练 =====
for epoch in range(1,EPOCHS+1):
    tot,lu_acc,ls_acc,lp_acc=0,0,0,0
    for g in graphs:
        out_u,out_s,out_p=model(g["X"],g["edge"])
        lu=F.mse_loss(out_u,g["Y_u"])
        ls=F.mse_loss(out_s,g["Y_s"])
        lp=F.mse_loss(out_p,g["Y_p"])
        loss=lu + W_S*ls + W_P*lp
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM)
        opt.step()
        tot+=loss.item(); lu_acc+=lu.item(); ls_acc+=ls.item(); lp_acc+=lp.item()
    if epoch==1 or epoch%10==0:
        print(f"Ep{epoch:04d} tot={tot:.4f}  U={lu_acc:.4f}  S={ls_acc:.4f}  P={lp_acc:.4f}")

# ===== 6. 保存模型 =====
torch.save({
    "model":model.state_dict(),
    "mu_x":mu_x, "std_x":std_x,
    "mu_u":mu_u, "std_u":std_u,
    "mu_s":mu_s, "std_s":std_s,
    "mu_p":mu_p, "std_p":std_p
}, SAVE_PATH)
print("Model saved to",SAVE_PATH)
