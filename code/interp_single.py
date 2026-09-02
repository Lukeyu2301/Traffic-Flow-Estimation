# ============================================================
# 绘制 Waterfall 图（修正逻辑与路径对齐版）
# ============================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import joblib

# ============================================================
# 1. 路径设置 (请确保 seed 和时段对齐)
# ============================================================
seed = 42
period = "morning" # 可选 morning/noon/evening
save_dir = rf"results\interp\{period}\seed_{seed}"
xgb_dir = os.path.join(save_dir, "for_r")
r_data_dir = os.path.join(xgb_dir, "r_shapr_data")
results_dir = os.path.join(r_data_dir, "results")

# 加载之前保存的 Scaler 和特征清单
scaler_path = os.path.join(save_dir, "scalers", "scaler_full.pkl")
# 如果没有 feature_columns.json，我们现场从 X_test 获取
x_test_path = os.path.join(r_data_dir, "X_test.csv")

# ============================================================
# 2. 特征名称映射
# ============================================================
name_mapping = {
    "fclass": "RC",
    "popden": "POP",
    "closeness": "CC",
    "betweenness": "BC",
    "node_degree": "Degree",
    "len": "Length",
    "graph_embedding": "RWA",
    "交通设施": "TRA",
    "住宿住宅": "ACC",
    "公司企业": "COM",
    "科教文化": "CUL",
    "餐饮美食": "CAT",
    "休闲娱乐": "ENT",
    "生活服务": "LIV",
    "购物消费": "SHO"
}

categorical_features = ['fclass']

# ============================================================
# 3. 读取数据 (修复逻辑)
# ============================================================
print(">>> 正在读取分析数据...")

# 3.1 读取 SHAP 值
shap_values_df = pd.read_csv(os.path.join(results_dir, "shap_values_grouped.csv"))
shap_feature_cols = [col for col in shap_values_df.columns if col not in ['none', 'id', 'pred', 'explain_id']]

# 3.2 读取 phi0 (基准值)
with open(os.path.join(r_data_dir, "group_info.json"), 'r') as f:
    phi0 = json.load(f)['phi0']

# 3.3 读取采样索引
sample_indices = pd.read_csv(os.path.join(results_dir, "sample_indices.csv"))['sample_idx'].values

# 3.4 关键：读取/创建 test_info.csv (关联路段 ID)
pred_best_path = os.path.join(xgb_dir, "predictions_best.csv")
test_info_path = os.path.join(r_data_dir, "test_info.csv")

if os.path.exists(test_info_path):
    test_info_df = pd.read_csv(test_info_path)
    print("✅ 已读取现有的 test_info.csv")
elif os.path.exists(pred_best_path):
    pred_best_df = pd.read_csv(pred_best_path)
    test_info_df = pd.DataFrame({
        'row_idx': np.arange(len(pred_best_df)),
        'mapped_id': pred_best_df['mapped_id'].values if 'mapped_id' in pred_best_df.columns else np.arange(len(pred_best_df)),
        'y_true': pred_best_df['y_true'].values,
        'y_pred_python': pred_best_df['y_pred'].values
    })
    test_info_df.to_csv(test_info_path, index=False)
    print("✅ 已根据 predictions_best.csv 创建 test_info.csv")
else:
    # 如果实在找不到，创建一个临时的匿名 ID 表
    X_test_temp = pd.read_csv(x_test_path)
    test_info_df = pd.DataFrame({'mapped_id': np.arange(len(X_test_temp))})
    print("⚠️ 未找到预测结果文件，已生成临时路段 ID")

# 3.5 加载 Scaler
scaler_full = joblib.load(scaler_path)
X_test_df = pd.read_csv(x_test_path)
feature_names_in_order = list(X_test_df.columns)

# ============================================================
# 4. 创建 SHAP 样本对应表
# ============================================================
shap_sample_info = []
for shap_idx in range(len(shap_values_df)):
    r_idx = sample_indices[shap_idx]
    p_idx = r_idx - 1 # Python 0-based
    
    if p_idx < len(test_info_df):
        row = test_info_df.iloc[p_idx]
        shap_sample_info.append({
            'shap_idx': shap_idx,
            'python_idx': p_idx,
            'mapped_id': row.get('mapped_id', p_idx),
            'y_true': row.get('y_true', 0),
            'y_pred_r': phi0 + shap_values_df.iloc[shap_idx][shap_feature_cols].sum()
        })

shap_sample_df = pd.DataFrame(shap_sample_info)

# ============================================================
# 打印所有可供查询的采样路段 ID
# ============================================================
if not shap_sample_df.empty:
    # 提取去重并排序后的 ID
    available_ids = sorted(shap_sample_df['mapped_id'].dropna().unique().astype(int))
    
    print("\n" + "="*60)
    print(f"📊 可用于绘制 Waterfall 图的采样路段列表 (总计: {len(available_ids)} 个)")
    print("="*60)
    
    # 每行打印 10 个 ID，方便查看
    for i in range(0, len(available_ids), 10):
        line = available_ids[i:i+10]
        print(", ".join(f"{x:4d}" for x in line))
    
    print("="*60)
    
    # 额外提示预测偏差最大的路段 (可选)
    shap_sample_df['error'] = (shap_sample_df['y_true'] - shap_sample_df['y_pred_r']).abs()
    worst_road = shap_sample_df.loc[shap_sample_df['error'].idxmax()]
    print(f"💡 提示：预测误差最大的路段 ID 是 {int(worst_road['mapped_id'])} (误差: {worst_road['error']:.2f})")
    print(f"💡 提示：你可以尝试将 target_id 设为以上列表中的任意数字。")
else:
    print("❌ 对应表为空，无法获取路段 ID。请检查 predictions_best.csv 是否包含 mapped_id 列。")

# ============================================================
# 5. Waterfall 绘图函数 (保持你的核心逻辑)
# ============================================================
def plot_waterfall(shap_values, feature_names, feature_values, phi0, prediction, mapped_id, y_true, output_path):
    plt.rcParams['font.family'] = 'Arial'
    
    # 构造 DataFrame 并排序
    df = pd.DataFrame({'feature': feature_names, 'shap_value': shap_values})
    df['feat_val'] = df['feature'].apply(lambda x: feature_values.get(x, None))
    df['abs_shap'] = df['shap_value'].abs()
    df = df.sort_values('abs_shap', ascending=False).head(12) # 取前 12 个
    df = df.sort_values('shap_value', ascending=True)
    
    # 格式化特征显示名字
    def format_label(row):
        name = name_mapping.get(row['feature'], row['feature'])
        val = row['feat_val']
        if val is None or row['feature'] == 'graph_embedding': return name
        return f"{val:.2f} = {name}" if isinstance(val, float) else f"{val} = {name}"

    labels = df.apply(format_label, axis=1)
    
    # 计算 Waterfall 阶梯坐标
    n = len(df)
    cumsum = np.zeros(n + 1)
    cumsum[0] = phi0
    for i in range(n): cumsum[i+1] = cumsum[i] + df.iloc[i]['shap_value']
    
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#ff0051" if x >= 0 else "#008bfb" for x in df['shap_value']]
    
    # 画柱子
    for i in range(n):
        ax.barh(i, df.iloc[i]['shap_value'], left=min(cumsum[i], cumsum[i+1]), color=colors[i], height=0.6)
        # 加标注
        txt_x = cumsum[i+1] + 0.05 if df.iloc[i]['shap_value'] > 0 else cumsum[i+1] - 0.05
        ax.text(txt_x, i, f"{df.iloc[i]['shap_value']:+.2f}", va='center', ha='left' if df.iloc[i]['shap_value']>0 else 'right')

    # 画连接线
    for i in range(n-1):
        ax.plot([cumsum[i+1], cumsum[i+1]], [i-0.3, i+1.3], color='gray', linestyle='--', lw=0.8, alpha=0.5)

    ax.axvline(phi0, color='black', lw=1, alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_title(f"SHAP Waterfall (Road ID: {int(mapped_id)})", fontweight='bold')
    ax.set_xlabel(f"E[f(x)]={phi0:.2f}  --->  f(x)={prediction:.2f} (True={y_true:.2f})")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()

# ============================================================
# 6. 执行绘图 (修正切片逻辑版)
# ============================================================
target_id = 64 # 你可以根据刚才打印出来的列表修改这个 ID
matches = shap_sample_df[shap_sample_df['mapped_id'] == target_id]

if not matches.empty:
    # 获取 Scaler 预期的特征数量 (应该是 14)
    n_original_features = scaler_full.n_features_in_
    
    for _, row in matches.iterrows():
        # 1. 提取 SHAP (这部分没问题，维持 46 维)
        s_vals = shap_values_df.iloc[int(row['shap_idx'])][shap_feature_cols].values
        
        # 2. 还原特征值 (关键修复点：只对前 N 列进行反标准化)
        # 获取该样本全量特征 (46维)
        full_sc = X_test_df.iloc[int(row['python_idx'])].values.reshape(1, -1)
        
        # 【修改处】只取前 n_original_features 列进行还原
        raw_sc_base = full_sc[:, :n_original_features]
        raw_orig_base = scaler_full.inverse_transform(raw_sc_base)[0]
        
        # 3. 构造特征字典
        # 原始特征部分
        f_vals = dict(zip(feature_names_in_order[:n_original_features], raw_orig_base))
        # Embedding 部分 (不需要反标准化，设为 None 或者保留原值)
        f_vals['graph_embedding'] = None 
        
        # 4. 绘图
        out_name = os.path.join(xgb_dir, f"waterfall_road_{target_id}.png")
        plot_waterfall(s_vals, shap_feature_cols, f_vals, phi0, row['y_pred_r'], target_id, row['y_true'], out_name)
        
        print(f"✅ 已成功生成路段 {target_id} 的 Waterfall 图: {out_name}")
else:
    print(f"❌ 找不到路段 {target_id} 的采样数据。")
