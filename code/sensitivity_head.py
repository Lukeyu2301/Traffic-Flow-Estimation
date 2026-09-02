import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd
import optuna
import json
import joblib
from torch_geometric.nn import GATv2Conv, BatchNorm
from torch_geometric.utils import negative_sampling, to_undirected
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
import xgboost as xgb

# ================= 1. 环境配置 =================
SEED = 42
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路径设置
road_path = r'D:\KEKE\宣城交通流量预测\data\road_final_features.csv'
traffic_path = r'D:\KEKE\宣城交通流量预测\data\Morning_Average.csv' 
root_save_dir = rf"results\interp\morning\seed_{SEED}\sensitivity_analysis"
os.makedirs(root_save_dir, exist_ok=True)

# ================= 2. 动态模型定义 =================
class GATEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers, heads, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # 第一层
        self.convs.append(GATv2Conv(in_c, hid_c, heads=heads, edge_dim=1, dropout=dropout))
        self.bns.append(BatchNorm(hid_c * heads))
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hid_c * heads, hid_c, heads=heads, edge_dim=1, dropout=dropout))
            self.bns.append(BatchNorm(hid_c * heads))
        # 最后一层 (heads固定为1以输出固定维度的embedding)
        self.convs.append(GATv2Conv(hid_c * heads if num_layers > 1 else in_c, out_c, heads=1, edge_dim=1, dropout=dropout))
        self.proj = nn.Sequential(nn.Linear(out_c, out_c), nn.ReLU(), nn.Linear(out_c, out_c))

    def forward(self, x, edge_index, edge_attr):
        for i in range(len(self.convs) - 1):
            x = F.elu(self.bns[i](self.convs[i](x, edge_index, edge_attr)))
        z = self.convs[-1](x, edge_index, edge_attr)
        return z, self.proj(z)

def compute_all_metrics(y_true_sc, y_pred_sc, scaler_y):
    y_t = np.expm1(scaler_y.inverse_transform(y_true_sc.reshape(-1, 1)).flatten())
    y_p = np.maximum(np.expm1(scaler_y.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()), 0)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    mape = np.mean(np.abs((y_t - y_p) / (y_t + 1e-9))) * 100
    return {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}

# ================= 3. 数据准备 =================
print(">>> 正在初始化数据...")
df_raw = pd.read_csv(road_path, encoding='gbk')
df_traffic = pd.read_csv(traffic_path)
data_all = df_raw.merge(df_traffic, left_on='cid', right_on='ROAD_ID')
exclude = ['cid', 'ROAD_ID', 'avg_flow', 'geometry', 'ROAD_NAME', 'Unnamed: 0']
feature_cols = [c for c in data_all.columns if c not in exclude]
if 'fclass' in data_all.columns:
    data_all['fclass'] = LabelEncoder().fit_transform(data_all['fclass'].astype(str))

X_sc = StandardScaler().fit_transform(data_all[feature_cols].values)
scaler_y = StandardScaler()
Y_sc = scaler_y.fit_transform(np.log1p(data_all['avg_flow'].values).reshape(-1, 1)).flatten()

# 图构建
node_to_roads = {}
for idx, cid in enumerate(data_all['cid']):
    for n in str(cid).split('_'): node_to_roads.setdefault(n, []).append(idx)
edge_from, edge_to = [], []
for roads in node_to_roads.values():
    for r1 in roads:
        for r2 in roads:
            if r1 != r2: edge_from.append(r1); edge_to.append(r2)
edge_index = to_undirected(torch.tensor([edge_from, edge_to], dtype=torch.long)).to(device)
x_tensor = torch.tensor(X_sc, dtype=torch.float).to(device)
edge_attr = ((F.cosine_similarity(x_tensor[edge_index[0]], x_tensor[edge_index[1]], dim=1) + 1) / 2).view(-1, 1)

# ================= 4. 敏感性实验循环 =================
heads_to_test = [1, 2, 3, 4, 5, 6]
sensitivity_results = []

for head_count in heads_to_test:
    print(f"\n" + "="*60 + f"\n>>> 正在测试注意力头数 (Heads): {head_count}")
    
    # --- 1. 内部优化 GAT 参数 (此时 heads 已固定) ---
    def gat_obj(trial):
        hc = trial.suggest_int('hid', 32, 128, step=32)
        nl = trial.suggest_int('num_layers', 2, 4)
        lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
        tmp = trial.suggest_float('temp', 0.1, 0.4)
        m = GATEncoder(X_sc.shape[1], hc, 32, nl, head_count, 0.2).to(device)
        optimizer = torch.optim.Adam(m.parameters(), lr=lr)
        for _ in range(100):
            optimizer.zero_grad()
            _, p = m(x_tensor, edge_index, edge_attr)
            p_n = F.normalize(p, p=2, dim=1)
            pos = torch.sum(p_n[edge_index[0]] * p_n[edge_index[1]], dim=-1) / tmp
            neg_e = negative_sampling(edge_index, num_nodes=X_sc.shape[0])
            neg = torch.sum(p_n[neg_e[0]] * p_n[neg_e[1]], dim=-1) / tmp
            loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8)).mean()
            loss.backward(); optimizer.step()
        return loss.item()

    study_g = optuna.create_study(direction='minimize')
    study_g.optimize(gat_obj, n_trials=5)
    best_g = study_g.best_params
    
    # 提取正式 Embedding
    final_enc = GATEncoder(X_sc.shape[1], best_g['hid'], 32, best_g['num_layers'], head_count, 0.2).to(device)
    emb = final_enc(x_tensor, edge_index, edge_attr)[0].detach().cpu().numpy()
    X_comb = np.hstack([X_sc, emb])

    # --- 2. 5 折交叉验证评估并优化 XGBoost ---
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_comb)):
        xtr, xvl = X_comb[train_idx], X_comb[val_idx]
        ytr, yvl = Y_sc[train_idx], Y_sc[val_idx]

        def xgb_obj(trial):
            param = {
                'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'tree_method': 'hist', 'random_state': SEED
            }
            m = xgb.XGBRegressor(**param).fit(xtr, ytr)
            return mean_squared_error(yvl, m.predict(xvl))

        study_x = optuna.create_study(direction='minimize')
        study_x.optimize(xgb_obj, n_trials=8)
        
        m_final = xgb.XGBRegressor(**study_x.best_params).fit(xtr, ytr)
        res = compute_all_metrics(yvl, m_final.predict(xvl), scaler_y)
        fold_metrics.append(res)
    
    # 记录该 Head 数量下的平均结果
    avg_res = pd.DataFrame(fold_metrics).mean().to_dict()
    std_res = pd.DataFrame(fold_metrics).std().to_dict()
    
    combined_res = {"Heads": head_count}
    for k in avg_res:
        combined_res[f"{k}_mean"] = avg_res[k]
        combined_res[f"{k}_std"] = std_res[k]
    
    sensitivity_results.append(combined_res)
    print(f"Heads {head_count} 测试完成: RMSE={avg_res['RMSE']:.2f}, R2={avg_res['R2']:.4f}")

# ================= 5. 保存结果 =================
final_df = pd.DataFrame(sensitivity_results)
save_path = os.path.join(root_save_dir, "heads_sensitivity_results_1.csv")
final_df.to_csv(save_path, index=False)
print(f"\n✅ 敏感性分析完成！结果保存至: {save_path}")

# 简单的结果展示
print(final_df[["Heads", "RMSE_mean", "R2_mean", "MAE_mean"]])
