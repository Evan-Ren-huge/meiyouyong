#!/usr/bin/env python3
# view_pred_scatter.py  ——  节点散点可视化 (PyVista)
# -------------------------------------------------------------
import os, numpy as np, pyvista as pv

# ========== 1. 手动参数设置 ==========
args = lambda: None
args.npz    = r"D:\BaiduNetdiskDownload\npzpred\Job-t7_pred_speed.npz"
args.scalar = "U_mag"      # 可选: U_mag / Ux / Uy / Uz / pred_s / pred_peeq
args.frame  = 10          # 选择帧号（-1 表示最后一帧）
args.scale  = 5.0         # 形变放大倍数

if not os.path.isfile(args.npz):
    raise FileNotFoundError(args.npz)

# ========== 2. 读取并准备数据 ==========
data  = np.load(args.npz)
coord = data["node_coords"].astype(np.float32)    # (N,3)
conn  = data["connectivity"].astype(np.int32)     # (E,8)

# 处理位移和标量场
T = None
if args.scalar.startswith("U"):
    pred_u = data["pred_u"]                       # (T,N,3)
    T = pred_u.shape[0]
    fid = args.frame if args.frame>=0 else T+args.frame
    U   = pred_u[fid]
    # 标量：向量模或分量
    if args.scalar == "U_mag":
        scalars = np.linalg.norm(U, axis=1)
    else:
        comp = dict(Ux=0, Uy=1, Uz=2)[args.scalar]
        scalars = U[:,comp]
    # 变形坐标
    coords_def = coord + U * args.scale
else:
    arr = data[args.scalar]                       # (T,N) or (N,)
    if arr.ndim==2:
        T   = arr.shape[0]
        fid = args.frame if args.frame>=0 else T+args.frame
        scalars   = arr[fid]
    else:
        scalars = arr
    coords_def = coord

# ========== 3. 绘制散点图 ==========
# ========== 3. 绘制“方块”散点图 ==========
# 先构造一个只有点的 PolyData，并挂上标量
point_cloud = pv.PolyData(coords_def)
point_cloud.point_data[args.scalar] = scalars

# 生成一个单位立方体几何体（edge length=1）
cube = pv.Cube(x_length=1, y_length=1, z_length=1)

# 在每个节点位置放置一个立方体 glyph：
#  - scale=False 表示不按标量自动缩放
#  - factor 控制立方体实际大小（必须根据你的模型单位手动调节）
glyphs = point_cloud.glyph(
    geom=cube,
    scale=False,
    factor=24   # ← 根据需要调小或调大，0.05 只是示例
)

plotter = pv.Plotter(window_size=(900,700))
plotter.add_mesh(
    glyphs,
    scalars=args.scalar,
    cmap="turbo",
    scalar_bar_args={"title": args.scalar}
)
plotter.add_axes()
if T is not None:
    plotter.add_text(f"frame {fid} / {T-1}", 10)
plotter.show()
