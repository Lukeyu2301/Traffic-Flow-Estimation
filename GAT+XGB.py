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
from sklearn.model_selection import KFold
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
traffic_path = r'D:\KEKE\宣城交通流量预测\data\Evening_Average.csv'
root_save_dir = rf"results\interp\evening\seed_{SEED}"
xgb_dir = os.path.join(root_save_dir, "for_r")
r_data_dir = os.path.join(xgb_dir, "r_shapr_data")
full_data_check_dir = os.path.join(root_save_dir, "full_dataset_check") # 全量数据检查目录

os.makedirs(r_data_dir, exist_ok=True)
os.makedirs(full_data_check_dir, exist_ok=True)
os.makedirs(os.path.join(root_save_dir, "scalers"), exist_ok=True)

# ================= 2. 数据加载与动态特征提取 =================
print(">>> 正在加载数据并动态提取特征列...")
df_raw = pd.read_csv(road_path, encoding='gbk')
df_traffic = pd.read_csv(traffic_path)
data_all = df_raw.merge(df_traffic, left_on='cid', right_on='ROAD_ID')

# 自动确定特征列：排除非特征列
exclude_list = ['cid', 'ROAD_ID', 'avg_flow', 'geometry', 'ROAD_NAME', 'Unnamed: 0']
feature_cols = [c for c in data_all.columns if c not in exclude_list]

print(f"检测到特征数量: {len(feature_cols)}")

if 'fclass' in data_all.columns:
    data_all['fclass'] = LabelEncoder().fit_transform(data_all['fclass'].astype(str))

X_raw = data_all[feature_cols].values
Y_raw = data_all['avg_flow'].values

# 标准化
scaler_x = StandardScaler()
X_sc = scaler_x.fit_transform(X_raw)
scaler_y = StandardScaler()
Y_log = np.log1p(Y_raw)
Y_sc = scaler_y.fit_transform(Y_log.reshape(-1, 1)).flatten()

# 保存 Scaler
joblib.dump(scaler_x, os.path.join(root_save_dir, "scalers", "scaler_full.pkl"))

# ================= 3. 构建无向图 (负采样修正) =================
print(">>> 正在构建无向图结构...")
edge_from, edge_to = [], []
node_to_roads = {}
for idx, cid in enumerate(data_all['cid']):
    for n in str(cid).split('_'): node_to_roads.setdefault(n, []).append(idx)
for roads in node_to_roads.values():
    for r1 in roads:
        for r2 in roads:
            if r1 != r2: edge_from.append(r1); edge_to.append(r2)

edge_index_raw = torch.tensor([edge_from, edge_to], dtype=torch.long)
edge_index = to_undirected(edge_index_raw).to(device) # 转为双向边，修正负采样
x_tensor = torch.tensor(X_sc, dtype=torch.float).to(device)
edge_attr = ((F.cosine_similarity(x_tensor[edge_index[0]], x_tensor[edge_index[1]], dim=1) + 1) / 2).view(-1, 1)


# ================= 4. GAT 参数优化 (InfoNCE) =================
class GATEncoder(nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_layers=2, heads=4, dropout=0.1):
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # 第一层
        self.convs.append(GATv2Conv(in_c, hid_c, heads=heads, edge_dim=1, dropout=dropout))
        self.bns.append(BatchNorm(hid_c * heads))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hid_c * heads, hid_c, heads=heads, edge_dim=1, dropout=dropout))
            self.bns.append(BatchNorm(hid_c * heads))
        
        # 最后一层
        self.convs.append(GATv2Conv(hid_c * heads, out_c, heads=1, edge_dim=1, dropout=dropout))
        
        self.proj = nn.Sequential(
            nn.Linear(out_c, out_c), 
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(out_c, out_c)
        )

    def forward(self, x, edge_index, edge_attr):
        # 前 num_layers-1 层
        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.elu(self.bns[i](x))
        
        # 最后一层（无激活）
        z = self.convs[-1](x, edge_index, edge_attr)
        return z, self.proj(z)

def gat_objective(trial):
    # ✅ 添加所有超参数
    num_layers = trial.suggest_int('num_layers', 2, 4)
    hid = trial.suggest_int('hid', 32, 128, step=32)
    heads = trial.suggest_categorical('heads', [2, 4, 8])
    dropout = trial.suggest_float('dropout', 0.1, 0.4)
    lr = trial.suggest_float('lr', 1e-4, 5e-3, log=True)
    temp = trial.suggest_float('temp', 0.1, 0.4)
    
    model = GATEncoder(
        X_sc.shape[1], 
        hid, 
        32, 
        num_layers=num_layers,
        heads=heads,
        dropout=dropout
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for _ in range(200):
        optimizer.zero_grad()
        z, p = model(x_tensor, edge_index, edge_attr)
        p_n = F.normalize(p, p=2, dim=1)
        
        pos = torch.sum(p_n[edge_index[0]] * p_n[edge_index[1]], dim=-1) / temp
        neg_e = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1))
        neg = torch.sum(p_n[neg_e[0]] * p_n[neg_e[1]], dim=-1) / temp
        
        loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8)).mean()
        loss.backward()
        optimizer.step()
    
    return loss.item()

print(">>> 优化 GAT...")
study_gat = optuna.create_study(direction='minimize')
study_gat.optimize(gat_objective, n_trials=15)  # 增加trial数以覆盖更多组合
best_gat_p = study_gat.best_params
print(f"最佳 GAT 参数: {best_gat_p}")

# 最终提取 Embedding (使用最佳参数)
final_gat = GATEncoder(
    X_sc.shape[1], 
    best_gat_p['hid'], 
    32, 
    num_layers=best_gat_p['num_layers'],
    heads=best_gat_p['heads'],
    dropout=best_gat_p['dropout']
).to(device)

optimizer_g = torch.optim.Adam(final_gat.parameters(), lr=best_gat_p['lr'])

print(">>> 训练最终 GAT...")
for epoch in range(401):
    optimizer_g.zero_grad()
    z, p = final_gat(x_tensor, edge_index, edge_attr)
    p_n = F.normalize(p, p=2, dim=1)
    
    pos = torch.sum(p_n[edge_index[0]] * p_n[edge_index[1]], dim=-1) / best_gat_p['temp']
    neg_e = negative_sampling(edge_index, num_nodes=z.size(0), num_neg_samples=edge_index.size(1))
    neg = torch.sum(p_n[neg_e[0]] * p_n[neg_e[1]], dim=-1) / best_gat_p['temp']
    
    loss = -torch.log(torch.exp(pos) / (torch.exp(pos) + torch.exp(neg) + 1e-8)).mean()
    loss.backward()
    optimizer_g.step()
    
    if epoch % 100 == 0:
        print(f"  Epoch {epoch:3d}/400 | Loss: {loss.item():.4f}")

spatial_emb = final_gat(x_tensor, edge_index, edge_attr)[0].detach().cpu().numpy()
X_combined = np.hstack([X_sc, spatial_emb])
emb_cols = [f"emb_{i}" for i in range(32)]
all_cols = feature_cols + emb_cols

# ================= 5. XGBoost 5折交叉验证并按类别统计 =================
print("\n>>> 开始 5 折交叉验证...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_results = []
all_fold_grouped_metrics = [] # 新增：用于存储每一折的分组指标
best_rmse_val = float('inf')
best_fold_bundle = None

for fold, (train_idx, val_idx) in enumerate(kf.split(X_combined)):
    xtr, xvl = X_combined[train_idx], X_combined[val_idx]
    ytr, yvl = Y_sc[train_idx], Y_sc[val_idx]
    
    def xgb_obj(trial):
        ps = {'n_estimators': trial.suggest_int('n_estimators', 200, 2000), 'max_depth': trial.suggest_int('max_depth', 3, 12),
              'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True), 'subsample': trial.suggest_float('subsample', 0.6, 0.9),
              'tree_method': 'hist', 'random_state': SEED}
        m = xgb.XGBRegressor(**ps); m.fit(xtr, ytr)
        return mean_squared_error(yvl, m.predict(xvl))

    study_xgb = optuna.create_study(direction='minimize')
    study_xgb.optimize(xgb_obj, n_trials=20)
    
    m_fold = xgb.XGBRegressor(**study_xgb.best_params); m_fold.fit(xtr, ytr)
    
    # 预测并还原
    preds_raw = m_fold.predict(xvl)
    preds_inv = np.maximum(np.expm1(scaler_y.inverse_transform(preds_raw.reshape(-1,1)).flatten()), 0)
    true_inv = np.expm1(scaler_y.inverse_transform(yvl.reshape(-1,1)).flatten())
    
    # 全局指标计算
    f_rmse = np.sqrt(mean_squared_error(true_inv, preds_inv))
    f_r2 = r2_score(true_inv, preds_inv)
    f_mae = mean_absolute_error(true_inv, preds_inv)
    f_mape = np.mean(np.abs((true_inv - preds_inv) / (true_inv + 1e-9))) * 100
    
    cv_results.append({'Fold': fold, 'RMSE': f_rmse, 'MAE': f_mae, 'R2': f_r2, 'MAPE': f_mape})
    print(f"Fold {fold} | RMSE: {f_rmse:.2f} | R2: {f_r2:.4f}")
    
    # ✅ 新增：计算本折内各道路类型的指标
    fold_df = data_all.iloc[val_idx][['fclass']].copy()
    fold_df['actual'] = true_inv
    fold_df['predicted'] = preds_inv
    
    for f_class, group in fold_df.groupby('fclass'):
        y_t, y_p = group['actual'].values, group['predicted'].values
        all_fold_grouped_metrics.append({
            'Fold': fold,
            'fclass': f_class,
            'MAE': mean_absolute_error(y_t, y_p),
            'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
            'MAPE(%)': np.mean(np.abs((y_t - y_p) / (y_t + 1e-9))) * 100,
            'R2': r2_score(y_t, y_p) if len(y_t) > 1 else 0
        })

    if f_rmse < best_rmse_val:
        best_rmse_val = f_rmse
        best_fold_bundle = {'model': m_fold, 'params': study_xgb.best_params, 'xtr': xtr, 'xvl': xvl, 'ytr': ytr}

# ================= 新增：汇总 5 折分组指标均值 =================
print("\n>>> 正在计算 5 折道路类型汇总指标...")
all_groups_df = pd.DataFrame(all_fold_grouped_metrics)

# 按照 fclass 分组并计算所有 Fold 的平均值
final_grouped_summary = all_groups_df.groupby('fclass').agg({
    'MAE': ['mean', 'std'],
    'RMSE': ['mean', 'std'],
    'MAPE(%)': ['mean', 'std'],
    'R2': ['mean', 'std']
}).reset_index()

# 展平列名 (例如 MAE_mean)
final_grouped_summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in final_grouped_summary.columns]

# 保存汇总结果
final_grouped_summary.to_csv(os.path.join(xgb_dir, "metrics_by_road_type_5fold_avg.csv"), index=False)
print(f"✅ 5折分类汇总指标已保存至: {os.path.join(xgb_dir, 'metrics_by_road_type_5fold_avg.csv')}")
print(final_grouped_summary[['fclass', 'MAE_mean', 'RMSE_mean', 'R2_mean']])

# ================= 6. 双重导出：全量检查 + 最好一折 =================
# 1. 导出全量数据集 (用于检查)
print("\n>>> 正在导出全量数据集用于检查...")
full_df_check = pd.DataFrame(X_combined, columns=all_cols)
full_df_check['target_scaled'] = Y_sc
full_df_check['avg_flow_raw'] = Y_raw
full_df_check.to_csv(os.path.join(full_data_check_dir, "full_dataset_combined.csv"), index=False)

# 2. 导出最好一折 (用于 R 解释)
print(f"\n>>> 导出最优折 (RMSE: {best_rmse_val:.4f}) 数据至 R 文件夹...")
pd.DataFrame(best_fold_bundle['xtr'], columns=all_cols).to_csv(os.path.join(r_data_dir, "X_train.csv"), index=False)
pd.DataFrame(best_fold_bundle['xvl'], columns=all_cols).to_csv(os.path.join(r_data_dir, "X_test.csv"), index=False)
pd.DataFrame({'target': best_fold_bundle['ytr']}).to_csv(os.path.join(r_data_dir, "y_train.csv"), index=False)

# 保存指标
pd.DataFrame(cv_results).to_csv(os.path.join(xgb_dir, "cv_metrics_all_folds.csv"), index=False)
pd.DataFrame(cv_results).agg(['mean', 'std']).drop('Fold', axis=1).to_csv(os.path.join(xgb_dir, "summary.csv"))

# 保存模型与清洗后的参数
best_fold_bundle['model'].save_model(os.path.join(r_data_dir, "xgb_model.json"))
clean_params = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in best_fold_bundle['params'].items()}
with open(os.path.join(r_data_dir, "xgb_params.json"), 'w') as f: json.dump(clean_params, f, indent=2)

# Group 信息导出
mapping = [{'feature': c, 'group': c} for c in feature_cols] + [{'feature': c, 'group': 'graph_embedding'} for c in emb_cols]
pd.DataFrame(mapping).to_csv(os.path.join(r_data_dir, "group_mapping.csv"), index=False)
with open(os.path.join(r_data_dir, "group_info.json"), 'w') as f:
    json.dump({"phi0": float(best_fold_bundle['ytr'].mean()), "total_groups": len(feature_cols) + 1}, f, indent=2)

print(f"✅ 任务完成！全量数据见: {full_data_check_dir}, R 解释数据见: {r_data_dir}")