import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import optuna
import random
import os
from torch_geometric.nn import GATv2Conv, BatchNorm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ================= 1. 固定随机种子 =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ================= 2. 数据加载与预处理 =================
print("正在加载数据...")
features_df = pd.read_csv(r'D:\KEKE\宣城交通流量预测\data\road_final_features.csv', encoding='gbk')
# 假设道路类型列名为 'fclass'，如果不是，请修改此处
flow_df = pd.read_csv(r'D:\KEKE\1203交通流量预测\data\Noon_Average.csv')
data_all = features_df.merge(flow_df, left_on='cid', right_on='ROAD_ID', how='inner')

# 提取特征
exclude_cols = ['cid', 'ROAD_ID', 'avg_flow']
feature_cols = [c for c in data_all.columns if c not in exclude_cols]
X = data_all[feature_cols].values
Y = data_all['avg_flow'].values

# 标准化处理
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
Y_log = np.log1p(Y)
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y_log.reshape(-1, 1))

x_tensor = torch.tensor(X_scaled, dtype=torch.float)
y_tensor = torch.tensor(Y_scaled, dtype=torch.float)

# 图结构构建
def build_edge_index(df):
    edge_index_from, edge_index_to = [], []
    node_to_roads = {}
    for idx, cid in enumerate(df['cid']):
        nodes = str(cid).split('_')
        for n in nodes:
            if n not in node_to_roads: node_to_roads[n] = []
            node_to_roads[n].append(idx)
    for n in node_to_roads:
        roads = node_to_roads[n]
        for r1 in roads:
            for r2 in roads:
                if r1 != r2:
                    edge_index_from.append(r1); edge_index_to.append(r2)
    return torch.tensor([edge_index_from, edge_index_to], dtype=torch.long)

edge_index = build_edge_index(data_all)
src, dst = edge_index[0], edge_index[1]
edge_attr = F.cosine_similarity(x_tensor[src], x_tensor[dst], dim=1).view(-1, 1)
edge_attr = (torch.clamp(edge_attr, min=0.01) + 1e-6).to(torch.float)

# ================= 3. 定义模型与指标 =================
class FinalGATv2Model(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, heads, dropout, num_layers):
        super(FinalGATv2Model, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads, edge_dim=1, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels * heads))
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=1, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels * heads))
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * heads, hidden_channels),
            torch.nn.LayerNorm(hidden_channels), torch.nn.ReLU(),
            torch.nn.Dropout(dropout), torch.nn.Linear(hidden_channels, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            identity = x if i > 0 else None
            x = conv(x, edge_index, edge_attr)
            x = F.elu(bn(x))
            if identity is not None and identity.shape == x.shape:
                x = x + identity
        return self.regressor(x)

def calculate_mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ================= 4. Optuna 调参 (保持不变) =================
def objective(trial):
    h_params = {
        'hidden_channels': trial.suggest_int('hidden_channels', 32, 128, step=16),
        'heads': trial.suggest_categorical('heads', [2, 4, 8]),
        'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.4),
        'num_layers': trial.suggest_int('num_layers', 2, 4)
    }

    idx = np.arange(x_tensor.size(0))
    t_idx, v_idx = train_test_split(idx, test_size=0.2, random_state=42)
    model = FinalGATv2Model(X.shape[1], h_params['hidden_channels'], h_params['heads'], 
                             h_params['dropout'], h_params['num_layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=h_params['lr'], weight_decay=1e-3)
    criterion = torch.nn.HuberLoss()
    bx, by = x_tensor.to(device), y_tensor.to(device)
    bei, bea = edge_index.to(device), edge_attr.to(device)
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        out = model(bx, bei, bea)
        loss = criterion(out[t_idx], by[t_idx])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(bx, bei, bea)[v_idx].cpu().numpy()
        y_p = np.expm1(y_scaler.inverse_transform(pred)).flatten()
        y_t = np.expm1(y_scaler.inverse_transform(by[v_idx].cpu().numpy())).flatten()
        return mean_absolute_error(y_t, np.maximum(y_p, 0))

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)
best = study.best_params

# ================= 5. 最终交叉验证 & 结果详细记录 =================
kf = KFold(n_splits=5, shuffle=True, random_state=42)
all_folds_loss = []
all_predictions_df = [] 
fold_metrics_list = [] # 新增：用于存储每一折的指标

print("\n开始最终 5 折交叉验证并记录详细预测值...")
for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(x_tensor.size(0)))):
    train_mask = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_mask = torch.tensor(val_idx, dtype=torch.long).to(device)
    
    model = FinalGATv2Model(X.shape[1], best['hidden_channels'], best['heads'], 
                             best['dropout'], best['num_layers']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best['lr'], weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=30)
    criterion = torch.nn.HuberLoss()
    
    bx, by = x_tensor.to(device), y_tensor.to(device)
    bei, bea = edge_index.to(device), edge_attr.to(device)
    
    fold_loss = []
    for epoch in range(601):
        model.train()
        optimizer.zero_grad()
        out = model(bx, bei, bea)
        loss = criterion(out[train_mask], by[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        fold_loss.append(loss.item())
        if epoch % 200 == 0:
            model.eval()
            with torch.no_grad():
                v_pred = model(bx, bei, bea)[val_mask].cpu().numpy()
                v_mae = mean_absolute_error(np.expm1(y_scaler.inverse_transform(by[val_mask].cpu().numpy())), 
                                          np.expm1(y_scaler.inverse_transform(v_pred)))
                scheduler.step(v_mae)

    all_folds_loss.append(fold_loss)

    # 评估并提取本折预测结果
    model.eval()
    with torch.no_grad():
        out_raw = model(bx, bei, bea)[val_mask].cpu().numpy()
        y_true_real = np.expm1(y_scaler.inverse_transform(by[val_mask].cpu().numpy())).flatten()
        y_pred_real = np.maximum(np.expm1(y_scaler.inverse_transform(out_raw)).flatten(), 0)
        
        # 计算该折指标
        f_rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
        f_mae = mean_absolute_error(y_true_real, y_pred_real)
        f_mape = calculate_mape(y_true_real, y_pred_real)
        f_r2 = r2_score(y_true_real, y_pred_real)
        
        fold_metrics_list.append({
            'Fold': fold + 1,
            'RMSE': f_rmse,
            'MAE': f_mae,
            'MAPE(%)': f_mape,
            'R2': f_r2
        })
        
        # 提取 metadata (cid 和 fclass)
        fold_info = data_all.iloc[val_idx][['cid', 'fclass']].copy()
        fold_info['actual'] = y_true_real
        fold_info['predicted'] = y_pred_real
        fold_info['fold'] = fold + 1
        all_predictions_df.append(fold_info)
        
        print(f"Fold {fold+1} 完成 | RMSE: {f_rmse:.2f} | MAE: {f_mae:.2f} | R2: {f_r2:.4f}")

# 合并所有预测结果
final_results_df = pd.concat(all_predictions_df, axis=0).reset_index(drop=True)

# ================= 6. 保存数据与分析 =================
output_dir = r'D:\KEKE\宣城交通流量预测\results'
if not os.path.exists(output_dir): os.makedirs(output_dir)

# 1. 每一折的详细指标汇总与 Summary
fold_summary_df = pd.DataFrame(fold_metrics_list)
# 计算均值和标准差
mean_row = fold_summary_df.mean(numeric_only=True).to_frame().T
mean_row['Fold'] = 'Average'
std_row = fold_summary_df.std(numeric_only=True).to_frame().T
std_row['Fold'] = 'Std Dev'

final_summary_df = pd.concat([fold_summary_df, mean_row, std_row], axis=0).reset_index(drop=True)
final_summary_df.to_csv(os.path.join(output_dir, 'GATv2_folds_summary.csv'), index=False, encoding='gbk')
print(f"\n[系统] 五折交叉验证汇总报告已保存至: GATv2_folds_summary.csv")

# 2. 保存详细预测 CSV
final_results_df.to_csv(os.path.join(output_dir, 'GATv2_all_predictions.csv'), index=False, encoding='gbk')

# 3. 按道路类型统计指标 (修正了原来的统计部分)
type_metrics = []
for road_type, group in final_results_df.groupby('fclass'):
    y_t, y_p = group['actual'].values, group['predicted'].values
    type_metrics.append({
        'Road_Type': road_type,
        'Count': len(group),
        'MAE': mean_absolute_error(y_t, y_p),
        'RMSE': np.sqrt(mean_squared_error(y_t, y_p)),
        'MAPE(%)': calculate_mape(y_t, y_p),
        'R2': r2_score(y_t, y_p)
    })
metrics_df = pd.DataFrame(type_metrics)
metrics_df.to_csv(os.path.join(output_dir, 'GATv2_metrics_by_type.csv'), index=False, encoding='gbk')

# 4. 打印最终全量汇总结果
print("\n" + "="*50)
print(f"{'GATv2 五折交叉验证最终平均指标':^40}")
print("-"*50)
print(f"平均 RMSE : {final_summary_df.loc[final_summary_df['Fold']=='Average', 'RMSE'].values[0]:.4f}")
print(f"平均 MAE  : {final_summary_df.loc[final_summary_df['Fold']=='Average', 'MAE'].values[0]:.4f}")
print(f"平均 MAPE : {final_summary_df.loc[final_summary_df['Fold']=='Average', 'MAPE(%)'].values[0]:.4f}%")
print(f"平均 R2   : {final_summary_df.loc[final_summary_df['Fold']=='Average', 'R2'].values[0]:.4f}")
print("="*50)
