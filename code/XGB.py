import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import random
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================= 1. 固定随机种子 =================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # XGBoost 内部随机性通过 random_state 参数控制

seed_everything(42)

# ================= 2. 加载与预处理数据 =================
print("正在加载数据...")
features_df = pd.read_csv(r'D:\KEKE\宣城交通流量预测\data\road_final_features.csv', encoding='gbk')
flow_df = pd.read_csv(r'D:\KEKE\1203交通流量预测\data\Morning_Average.csv')

data_all = features_df.merge(flow_df, left_on='cid', right_on='ROAD_ID', how='inner')

# 处理类别特征
le = LabelEncoder()
data_all['fclass'] = le.fit_transform(data_all['fclass'])

# 提取特征
exclude_cols = ['cid', 'ROAD_ID', 'avg_flow']
feature_cols = [c for c in data_all.columns if c not in exclude_cols]

X = data_all[feature_cols].values
Y = data_all['avg_flow'].values

# 特征标准化
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

# 目标变量对数转换 + 标准化
Y_log = np.log1p(Y)
y_scaler = StandardScaler()
Y_scaled = y_scaler.fit_transform(Y_log.reshape(-1, 1)).flatten()

def calculate_mape(y_true, y_pred):
    mask = y_true > 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# ================= 3. 定义 Optuna 调参目标函数 =================
def objective(trial):
    # 搜索空间
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42,
        'tree_method': 'hist',  # 使用直方图算法加速
        'n_jobs': -1
    }
    
    # 划分训练集和验证集 (与 GATv2 调参逻辑保持一致)
    X_train_tune, X_val_tune, y_train_tune, y_val_tune = train_test_split(
        X_scaled, Y_scaled, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(**param)
    
    # 训练并使用早停
    model.fit(
        X_train_tune, y_train_tune,
        eval_set=[(X_val_tune, y_val_tune)],
#         early_stopping_rounds=50,
        verbose=False
    )
    
    # 验证集预测
    preds = model.predict(X_val_tune)
    
    # 逆变换回真实物理单位
    y_pred_real = np.expm1(y_scaler.inverse_transform(preds.reshape(-1, 1)).flatten())
    y_true_real = np.expm1(y_scaler.inverse_transform(y_val_tune.reshape(-1, 1)).flatten())
    
    return mean_absolute_error(y_true_real, np.maximum(y_pred_real, 0))

# ================= 4. 执行 Optuna 搜索 =================
print("开始 XGBoost 参数优化...")
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50) # 尝试 50 组参数

print("\n" + "*"*30)
print("XGBoost 最优参数:", study.best_params)
print("*"*30)

# ================= 5. 使用最优参数进行最终 5 折交叉验证 =================
best_params = study.best_params
kf = KFold(n_splits=5, shuffle=True, random_state=42)
final_fold_results = []

print("\n开始最终 5 折交叉验证...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = Y_scaled[train_idx], Y_scaled[val_idx]
    
    # 使用 Optuna 找到的最佳参数
    model = xgb.XGBRegressor(**best_params, random_state=42, tree_method='hist')
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
#         early_stopping_rounds=50,
        verbose=False
    )
    
    # 预测并逆变换
    pred_scaled = model.predict(X_val)
    y_pred_log = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    y_true_log = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    y_pred_real = np.expm1(y_pred_log)
    y_true_real = np.expm1(y_true_log)
    y_pred_real = np.maximum(y_pred_real, 0)
    
    # 计算指标
    mae = mean_absolute_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_true_real, y_pred_real))
    mape = calculate_mape(y_true_real, y_pred_real)
    r2 = r2_score(y_true_real, y_pred_real)
    
    final_fold_results.append({'mae': mae, 'rmse': rmse, 'mape': mape, 'r2': r2})
    print(f"折 {fold+1} 结果: MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R2: {r2:.2f}")

# ================= 6. 输出汇总结果 =================
print("\n" + "="*40)
print("XGBoost + Optuna 5折交叉验证最终汇总:")
print(f"平均 MAE  : {np.mean([x['mae'] for x in final_fold_results]):.4f}")
print(f"平均 RMSE : {np.mean([x['rmse'] for x in final_fold_results]):.4f}")
print(f"平均 MAPE : {np.mean([x['mape'] for x in final_fold_results]):.2f}%")
print(f"平均 R2 : {np.mean([x['r2'] for x in final_fold_results]):.2f}")
print("="*40)

# 特征重要性分析
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print("\n最优模型特征重要性 (Top 5):")
for i in range(5):
    print(f"{i+1}. {feature_cols[indices[i]]}: {importances[indices[i]]:.4f}")
