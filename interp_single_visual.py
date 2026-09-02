# ============================================================
# Waterfall 图（全样本批量导出 - 横轴还原 + 细节优化版）
# ============================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import json
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# ================= 1. 路径与参数设置 =================
seed = 42
period = "morning"  
base_path = r'D:\KEKE\宣城交通流量预测'
save_dir = os.path.join(base_path, "results", "interp", period, f"seed_{seed}")
xgb_dir = os.path.join(save_dir, "for_r")
r_data_dir = os.path.join(xgb_dir, "r_shapr_data")
results_dir = os.path.join(r_data_dir, "results")
scaler_path = os.path.join(save_dir, "scalers", "scaler_full.pkl")

# 数据源
raw_flow_path = os.path.join(base_path, "data", f"{period.capitalize()}_Average.csv")
road_feat_path = os.path.join(base_path, "data", "road_final_features.csv")

output_folder = os.path.join(results_dir, "waterfall_plots_real_units_axis_0821")
os.makedirs(output_folder, exist_ok=True)

# ================= 2. 准备还原器 =================
print(">>> 正在准备流量还原器...")
df_f = pd.read_csv(road_feat_path, encoding='gbk')
df_t = pd.read_csv(raw_flow_path)
df_master = df_f.merge(df_t, left_on='cid', right_on='ROAD_ID')

scaler_y = StandardScaler()
y_log = np.log1p(df_master['avg_flow'].values)
scaler_y.fit(y_log.reshape(-1, 1))

def inv_y(val_sc):
    """将 Log-Standardized 数值还原为 辆/小时"""
    log_val = scaler_y.inverse_transform(np.array(float(val_sc)).reshape(-1, 1))[0, 0]
    return np.expm1(log_val)

# ================= 3. 加载数据 =================
shap_values_df = pd.read_csv(os.path.join(results_dir, "shap_values_grouped.csv"))
shap_feature_cols = [col for col in shap_values_df.columns if col not in ['none', 'id', 'pred', 'explain_id']]

with open(os.path.join(r_data_dir, "group_info.json"), 'r') as f:
    phi0_sc = json.load(f)['phi0']

sample_indices = pd.read_csv(os.path.join(results_dir, "sample_indices.csv"))['sample_idx'].values
X_test_df = pd.read_csv(os.path.join(r_data_dir, "X_test.csv"))
scaler_x = joblib.load(scaler_path)
n_orig = scaler_x.n_features_in_

# ================= 4. 绘图函数 (细节优化版) =================
def plot_waterfall_real_axis(shap_values, feature_values, phi0_sc, mapped_id, y_true_real, output_path):
    plt.rcParams['font.family'] = 'Arial'
    
    # 1. ✅ 修改点：展开所有特征，不再合并为 Others
    display_df = pd.DataFrame({'feature': shap_feature_cols, 'shap_value': shap_values})
    display_df['abs_shap'] = display_df['shap_value'].abs()
    # 按照绝对值贡献排序
    display_df = display_df.sort_values('abs_shap', ascending=True).reset_index(drop=True)
    
    # 2. 计算每一跳在真实流量空间的坐标
    n = len(display_df)
    path_sc = np.zeros(n + 1)
    path_sc[0] = phi0_sc
    for i in range(n):
        path_sc[i+1] = path_sc[i] + display_df.iloc[i]['shap_value']
    
    path_real = np.array([inv_y(x) for x in path_sc])
    phi0_real = path_real[0]
    pred_real = path_real[-1]
    
    # 3. 绘图设置
    x_min_data, x_max_data = min(path_real.min(), y_true_real), max(path_real.max(), y_true_real)
    x_range = x_max_data - x_min_data
    x_min = x_min_data - x_range * 0.4
    x_max = x_max_data + x_range * 0.2

    # ✅ 修改点：增加图表高度以适应所有特征展开，并控制垂直比例
    bar_spacing = 0.6  # 控制柱子之间的上下间距
    fig, ax = plt.subplots(figsize=(12, n * 1 + 2.5)) 
    ax.set_xlim(x_min, x_max)
    y_pos = np.arange(n) * bar_spacing # 应用间距
    
    name_mapping = {"fclass": "RC", "popden": "POP", "closeness": "CC", "betweenness": "BC", "node_degree": "Degree", "len": "Length", "graph_embedding": "RWA", "交通设施": "TRA", "住宿住宅": "ACC", "公司企业": "COM", "科教文化": "CUL", "餐饮美食": "CAT", "休闲娱乐": "ENT", "生活服务": "LIV", "购物消费": "SHO"}
    categorical_features = ['fclass', 'hour']

    for i in range(n):
        real_contrib = path_real[i+1] - path_real[i]
        start_real, end_real = path_real[i], path_real[i+1]
        
        color = "#FC0151" if real_contrib >= 0 else "#018AF9"
        y_mid = y_pos[i]
        y_top, y_bot = y_mid + 0.22, y_mid - 0.22
        
        head_len = min(x_range * 0.02, abs(real_contrib) * 0.5)
        
        if abs(real_contrib) > 1e-3:
            if real_contrib >= 0:
                verts = [(start_real, y_bot), (start_real, y_top), (end_real - head_len, y_top), (end_real, y_mid), (end_real - head_len, y_bot)]
            else:
                verts = [(start_real, y_bot), (start_real, y_top), (end_real + head_len, y_top), (end_real, y_mid), (end_real + head_len, y_bot)]
            ax.add_patch(Polygon(verts, closed=True, facecolor=color, edgecolor='none', zorder=3))
        
        text_ha = 'left' if real_contrib >= 0 else 'right'
        text_offset = x_range * 0.015 if real_contrib >= 0 else -x_range * 0.015
        
        if abs(real_contrib) > x_range * 0.25:
            ax.text((start_real + end_real)/2, y_mid, f"{real_contrib:+.1f}", va='center', ha='center', fontsize=42, color='white', fontweight='bold', zorder=4)
        else:
            ax.text(end_real + text_offset, y_mid, f"{real_contrib:+.1f}", va='center', ha=text_ha, fontsize=42, color=color, fontweight='bold', zorder=4)
            
        if i > 0:
            ax.plot([start_real, start_real], [y_pos[i-1], y_pos[i]], color='gray', linestyle='--', lw=1, alpha=0.3, zorder=1)

    # 4. ✅ 修改点：Y 轴标签还原及 CC/BC 四位小数处理
    labels = []
    for _, row in display_df.iterrows():
        fname = row['feature']
        f_display = name_mapping.get(fname, fname)
        f_val = feature_values.get(fname, None)
        
        if pd.notna(f_val) and fname != 'graph_embedding':
            # 针对 CC 和 BC 显示四位小数
            if fname in ['closeness', 'betweenness']:
                val_str = f"{f_val:.4f}"
            elif fname in categorical_features:
                val_str = f"{int(f_val)}"
            else:
                val_str = f"{f_val:.0f}" if f_val > 100 else f"{f_val:.2f}"
            
            labels.append(f"{val_str} = {f_display} ")
        else:
            labels.append(f"{f_display} ")
            
    # ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=44, fontweight='bold')
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=44)
    ax.tick_params(axis='x', labelsize=42, pad=10)
    
    # 5. 辅助线与标注
    # 1. 核心：紧凑设置 Y 轴范围，消除底部和顶部的多余留白
    # 底部下限设为第一个柱子中心往下 0.45 单位
    ax.set_ylim(y_pos[0] - 0.45, y_pos[-1] + 0.6)
    
    # 获取当前锁定后的坐标轴最底端位置（X 轴位置）
    y_ground = ax.get_ylim()[0]

    # 2. 绘制左侧 Baseline 灰色线
    # 位置使用 phi0_real，因为它对应你当前图表的 X 轴物理刻度
    ax.vlines(x=phi0_real, 
              ymin=y_ground, 
              ymax=y_pos[0] + 0.22, 
              color='gray', lw=1.2, alpha=0.5, zorder=2)

    # 3. 绘制右侧 Prediction 黑色线
    # 位置使用 pred_real，它对应最终预测的物理刻度
    # ax.vlines(x=pred_real, 
    #           ymin=y_ground, 
    #           ymax=y_pos[-1] + 0.22, 
    #           color='black', lw=1.8, zorder=5)
    ax.vlines(
        x=pred_real,
        ymin=y_ground,
        ymax=y_pos[-1] + 0.22,
        color='black',
        lw=1.3,          # 控制线宽，原来是 1.8
        # linestyles='--', # 改为虚线
        linestyles=(0, (8, 5)),   # (偏移量, [实线长度, 空白长度])
        zorder=5
    )

    # 4. 强制关闭自动留白
    ax.margins(y=0)
    
    # 注意：确保这里没有任何 ax.text(...) 语句，这样图表下方就是干净的 

    # ax.set_title(f"SHAP Waterfall (Road ID: {int(mapped_id)})", fontsize=16, fontweight='bold', pad=30)
    # ax.set_xlabel("Contribution to Estimation", fontsize=44, fontweight='bold')
    # ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
    # 设置底部 (X 轴)
    ax.spines['bottom'].set_color('black')
    ax.spines['bottom'].set_linewidth(2.5)

    # 设置左侧 (Y 轴)
    ax.spines['left'].set_color('black')
    ax.spines['left'].set_linewidth(2.5)

    # 如果你想隐藏顶部和右侧，但保持底和左的粗细
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close()

# ================= 5. 批量执行 =================
print(">>> 正在生成全展开还原后的 Waterfall 图表...")
feature_cols_names = list(X_test_df.columns)
master_matrix = df_master[feature_cols_names[:n_orig]].values

for i in tqdm(range(len(shap_values_df))):
    s_vals = shap_values_df.iloc[i][shap_feature_cols].values
    p_idx = sample_indices[i] - 1
    row_sc = X_test_df.iloc[p_idx, :n_orig].values.reshape(1, -1)
    row_raw = scaler_x.inverse_transform(row_sc)[0]
    
    dist = np.linalg.norm(master_matrix - row_raw, axis=1)
    match_row = df_master.iloc[np.argmin(dist)]
    
    f_vals = dict(zip(feature_cols_names[:n_orig], row_raw))
    f_vals['graph_embedding'] = None
    
    save_path = os.path.join(output_folder, f"ID_{int(match_row['cid'])}_Sample_{i}.png")
    plot_waterfall_real_axis(s_vals, f_vals, phi0_sc, match_row['cid'], match_row['avg_flow'], save_path)

print(f"\n✅ 任务完成！所有特征已展开，CC/BC 已设置为四位小数精度。")