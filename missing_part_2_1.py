import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import optuna
import joblib
import geopandas as gpd
from itertools import combinations
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tqdm import tqdm

# ================= 1. 环境配置 =================
SEED = 42
PERIOD = "noon"  # 可切换: morning / noon / evening

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 准备数据与 3 分区映射 =================
print(f">>> 正在加载 {PERIOD.upper()} 数据并读取外部预设 3 分区 SHP...")

base_path = r'D:\KEKE\宣城交通流量预测'
# 🟢 锁定 3 分区文件夹路径
blocks_dir = r'D:\KEKE\宣城交通流量预测\results\spatial_blocks_3'
road_path = os.path.join(base_path, "data", "road_final_features.csv")
traffic_path = os.path.join(base_path, "data", f"{PERIOD.capitalize()}_Average.csv")
base_save_dir = os.path.join(base_path, "results", "spatial_2vs1_optimized", PERIOD)
os.makedirs(base_save_dir, exist_ok=True)

# A. 加载特征与流量
df_raw = pd.read_csv(road_path, encoding='gbk')
df_traffic = pd.read_csv(traffic_path)
data_all = df_raw.merge(df_traffic, left_on='cid', right_on='ROAD_ID')

# B. 🔴 从外部 SHP 文件加载分区 (Block 0, 1, 2)
data_all['block'] = -1
for b_id in range(3):
    shp_file = os.path.join(blocks_dir, f"Spatial_Block_{b_id}.shp")
    if os.path.exists(shp_file):
        temp_gdf = gpd.read_file(shp_file)
        block_cids = temp_gdf['cid'].unique()
        data_all.loc[data_all['cid'].isin(block_cids), 'block'] = b_id
    else:
        print(f"⚠️ 警告：找不到分区文件 {shp_file}")

# 剔除未匹配路段
if (data_all['block'] == -1).any():
    data_all = data_all[data_all['block'] != -1].reset_index(drop=True)

print(f"✅ 分区加载完成。每块路段分布：\n{data_all['block'].value_counts().sort_index()}")

# C. 数据预处理
exclude = ['cid', 'ROAD_ID', 'avg_flow', 'geometry', 'block', 'Unnamed: 0']
feat_cols = [c for c in data_all.columns if c not in exclude]
if 'fclass' in data_all.columns:
    data_all['fclass'] = LabelEncoder().fit_transform(data_all['fclass'].astype(str))

X_sc_all = StandardScaler().fit_transform(data_all[feat_cols].values)
scaler_y = StandardScaler()
Y_sc_all = scaler_y.fit_transform(np.log1p(data_all['avg_flow'].values).reshape(-1, 1)).flatten()

# D. 图构建
node_to_roads = {}
for idx, cid in enumerate(data_all['cid']):
    for n in str(cid).split('_'): node_to_roads.setdefault(n, []).append(idx)
edge_from, edge_to = [], []
for roads in node_to_roads.values():
    for r1 in roads:
        for r2 in roads:
            if r1 != r2: edge_from.append(r1); edge_to.append(r2)

edge_index = to_undirected(torch.tensor([edge_from, edge_to], dtype=torch.long)).to(device)
x_tensor = torch.tensor(X_sc_all, dtype=torch.float).to(device)
edge_attr = ((F.cosine_similarity(x_tensor[edge_index[0]], x_tensor[edge_index[1]], dim=1) + 1) / 2).view(-1, 1).to(device)

# ================= 3. 模型定义 =================

class GATEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, heads, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATv2Conv(in_c, hid_c, heads=heads, edge_dim=1, dropout=dropout))
        self.bns.append(BatchNorm(hid_c * heads))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hid_c * heads, hid_c, heads=heads, edge_dim=1, dropout=dropout))
            self.bns.append(BatchNorm(hid_c * heads))
        self.convs.append(GATv2Conv(hid_c * heads, out_c, heads=1, edge_dim=1, dropout=dropout))
        self.proj = nn.Sequential(nn.Linear(out_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))
    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.convs)-1):
            x = F.elu(self.bns[i](self.convs[i](x, edge_index, edge_attr)))
        z = self.convs[-1](x, edge_index, edge_attr)
        return z, self.proj(z)

class GATMLPModel(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, heads, dropout):
        super().__init__()
        self.encoder = GATEncoder(in_c, hid_c, out_c, num_layers, heads, dropout)
        self.reg = nn.Sequential(nn.Linear(out_c, hid_c), nn.ReLU(), nn.Linear(hid_c, 1))
    def forward(self, x, edge_index, edge_attr):
        z, _ = self.encoder(x, edge_index, edge_attr)
        return self.reg(z)

def get_metrics_real(y_true_sc, y_pred_sc):
    y_t = np.expm1(scaler_y.inverse_transform(y_true_sc.reshape(-1, 1)).flatten())
    y_p = np.maximum(np.expm1(scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()), 0)
    return {"RMSE": np.sqrt(mean_squared_error(y_t, y_p)), "MAE": mean_absolute_error(y_t, y_p),
            "R2": r2_score(y_t, y_p), "MAPE": np.mean(np.abs((y_t - y_p) / (y_t + 1e-9))) * 100}

# ================= 4. 2-Blocks Train vs 1-Block Test 实验循环 =================
all_blocks = [0, 1, 2]
train_combinations = list(combinations(all_blocks, 2))
spatial_results = []

print(f"\n>>> 启动 2-Blocks 训练 vs 1-Block 测试 对比实验...")

for fold, train_tuple in enumerate(train_combinations):
    test_block = list(set(all_blocks) - set(train_tuple))[0]
    print(f"\n" + "="*60 + f"\n[Fold {fold+1}/3] 训练区域: {train_tuple} | 测试区域: Block {test_block}")
    
    train_idx_all = data_all[data_all['block'].isin(train_tuple)].index.values
    test_idx = data_all[data_all['block'] == test_block].index.values

    # --- A. GAT-XGBoost (双重调参) ---
    print("--- 正在调优 GAT-XGBoost ---")
    def gat_opt(trial):
        hc = trial.suggest_int('hid', 32, 128, step=32); nl = trial.suggest_int('num_layers', 2, 4)
        hd = trial.suggest_categorical('heads', [2, 4, 8]); lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        tmp = trial.suggest_float('temp', 0.1, 0.4)
        m = GATEncoder(X_sc_all.shape[1], hc, 32, nl, hd, 0.2).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(150):
            optimizer.zero_grad(); _, p = m(x_tensor, edge_index, edge_attr)
            p_n = F.normalize(p, p=2, dim=1)
            pos = torch.sum(p_n[edge_index[0]] * p_n[edge_index[1]], dim=-1) / tmp
            neg_e = negative_sampling(edge_index, num_nodes=X_sc_all.shape[0])
            neg = torch.sum(p_n[neg_e[0]] * p_n[neg_e[1]], dim=-1) / tmp
            loss = -torch.log(torch.exp(pos)/(torch.exp(pos)+torch.exp(neg)+1e-8)).mean()
            loss.backward(); optimizer.step()
        return loss.item()

    study_g = optuna.create_study(direction='minimize')
    study_g.optimize(gat_opt, n_trials=5)
    bg = study_g.best_params
    final_enc = GATEncoder(X_sc_all.shape[1], bg['hid'], 32, bg['num_layers'], bg['heads'], 0.2).to(device)
    emb = final_enc(x_tensor, edge_index, edge_attr)[0].detach().cpu().numpy()
    X_comb = np.hstack([X_sc_all, emb])

    def xgb_opt_logic(X_data, Y_data, t_idx):
        def obj(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'tree_method': 'hist', 'n_jobs': -1, 'random_state': SEED
            }
            # 在 2 个块内部划分 20% 验证
            xt, xv, yt, yv = train_test_split(X_data[t_idx], Y_data[t_idx], test_size=0.2, random_state=SEED)
            m = xgb.XGBRegressor(**param).fit(xt, yt)
            return mean_squared_error(yv, m.predict(xv))
        return obj

    # 模型拟合
    study_x1 = optuna.create_study(direction='minimize')
    study_x1.optimize(xgb_opt_logic(X_comb, Y_sc_all, train_idx_all), n_trials=10)
    m1 = xgb.XGBRegressor(**study_x1.best_params).fit(X_comb[train_idx_all], Y_sc_all[train_idx_all])
    res1 = get_metrics_real(Y_sc_all[test_idx], m1.predict(X_comb[test_idx]))
    res1.update({"Model": "GAT-XGBoost", "Fold": fold+1})
    spatial_results.append(res1)

    # --- B. 纯 XGBoost ---
    print("--- 正在优化 纯 XGBoost ---")
    study_x2 = optuna.create_study(direction='minimize')
    study_x2.optimize(xgb_opt_logic(X_sc_all, Y_sc_all, train_idx_all), n_trials=10)
    m2 = xgb.XGBRegressor(**study_x2.best_params).fit(X_sc_all[train_idx_all], Y_sc_all[train_idx_all])
    res2 = get_metrics_real(Y_sc_all[test_idx], m2.predict(X_sc_all[test_idx]))
    res2.update({"Model": "XGBoost", "Fold": fold+1})
    spatial_results.append(res2)

    # --- C. GAT-MLP ---
    print("--- 正在优化 GAT-MLP ---")
    def mlp_opt(trial):
        hc = trial.suggest_int('hid', 32, 128, step=16); hd = trial.suggest_categorical('heads', [2, 4, 8])
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True); nl = trial.suggest_int('num_layers', 2, 4); dr = trial.suggest_float('dropout', 0.1, 0.4)
        m = GATMLPModel(X_sc_all.shape[1], hc, 32, nl, hd, dr).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        xt, xv, yt, yv = train_test_split(train_idx_all, Y_sc_all[train_idx_all], test_size=0.2, random_state=SEED)
        yt_t = torch.tensor(Y_sc_all[xt], dtype=torch.float).view(-1,1).to(device)
        for _ in range(150):
            opt.zero_grad(); out = m(x_tensor, edge_index, edge_attr)
            F.mse_loss(out[xt], yt_t).backward(); opt.step()
        m.eval()
        with torch.no_grad(): p = m(x_tensor, edge_index, edge_attr)[xv].cpu().numpy()
        return mean_squared_error(Y_sc_all[xv], p)

    study_m = optuna.create_study(direction='minimize')
    study_m.optimize(mlp_opt, n_trials=10)
    bm = study_m.best_params
    m3 = GATMLPModel(X_sc_all.shape[1], bm['hid'], 32, bm['num_layers'], bm['heads'], bm['dropout']).to(device)
    opt3 = torch.optim.Adam(m3.parameters(), lr=bm['lr'])
    yt_t_all = torch.tensor(Y_sc_all[train_idx_all], dtype=torch.float).view(-1,1).to(device)
    for _ in range(300):
        opt3.zero_grad(); out = m3(x_tensor, edge_index, edge_attr)
        F.mse_loss(out[train_idx_all], yt_t_all).backward(); opt3.step()
    with torch.no_grad(): p3 = m3(x_tensor, edge_index, edge_attr)[test_idx].cpu().numpy().flatten()
    res3 = get_metrics_real(Y_sc_all[test_idx], p3)
    res3.update({"Model": "GAT-MLP", "Fold": fold+1})
    spatial_results.append(res3)

# ================= 5. 保存结果 =================
all_df = pd.DataFrame(spatial_results)
all_df.to_csv(os.path.join(base_save_dir, "spatial_2vs1_comparison_optimized.csv"), index=False)
summary = all_df.groupby("Model")[["RMSE", "MAE", "R2", "MAPE"]].agg(['mean', 'std'])
summary.to_csv(os.path.join(base_save_dir, "spatial_2vs1_summary_optimized.csv"))

print("\n" + "="*60 + f"\n📊 3 分区空间外推实验汇总 ({PERIOD.upper()})\n" + "="*60)
print(summary)