# ============================================================
# Python: 读取 R 的 shapr 结果并可视化 (标准化版本 - 双色对齐版)
# ============================================================
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from statsmodels.nonparametric.smoothers_lowess import lowess

print("\n" + "="*50)
print("读取 R shapr 结果并可视化 (使用标准化特征值)")
print("="*50)
seed = 42

# ================= 路径设置 =================
save_dir = rf"results\\interp\\evening\\seed_{seed}\\"
xgb_dir = os.path.join(save_dir, "for_r")
results_dir = os.path.join(xgb_dir, "r_shapr_data", "results")
nonlinear_dir = os.path.join(results_dir, "nonlinear_data")

# 特征名称映射
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

# 分类特征
categorical_features = ['fclass']

# ================= 1. 读取数据并绘制特征重要性柱状图 =================
# 定义道路属性分类（用于蓝色显示）
road_attributes = ["fclass", "popden", "closeness", "betweenness", "node_degree", "len", "graph_embedding"]

def plot_feature_importance(importance_df, output_path, top_k=15):
    top_df = importance_df.sort_values(by='importance_mean', ascending=False).head(top_k)
    
    original_features = top_df['feature_group'].values
    mapped_features = [name_mapping.get(f, f) for f in original_features]
    importances = top_df['importance_mean'].values
    
    # 配色逻辑：道路属性使用蓝色 (#EB9E56\913dd6\3b54e2)，POI 类型使用红色 (#FBD3A2\#D076F3\77adff)
    colors = ["#3b54e2" if f in road_attributes else "#82aef0" for f in original_features]
    
    y_positions = np.arange(len(mapped_features))

    plt.rcParams['font.family'] = 'Arial'
    fig, ax = plt.subplots(figsize=(4, 6))
    
    ax.barh(y_positions, importances, color=colors, height=0.8)
    ax.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(mapped_features, fontsize=14)
    ax.invert_yaxis()
    ax.tick_params(axis='x', labelsize=14)

    plt.savefig(output_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"✅ 特征重要性柱状图已保存 (双色版): {output_path}")

# --- 执行读取与重要性绘图 ---
importance_csv = os.path.join(results_dir, "feature_importance_grouped.csv")
if not os.path.exists(importance_csv):
    print(f"❌ 错误: 未找到文件 {importance_csv}，请确保 R 脚本已运行完成。")
    importance_df = None
else:
    importance_df = pd.read_csv(importance_csv)
    importance_plot_path = os.path.join(xgb_dir, "importance_bar_grouped.png")
    plot_feature_importance(importance_df, importance_plot_path)

# ================= 2. 绘制非线性依赖图 (前 5 个重要特征) =================
def plot_dependence_top5(importance_df, nonlinear_dir, output_path, top_k=5):
    if importance_df is None: return
    
    plt.rcParams['font.family'] = 'Arial'
    
    # 选取前 K 个重要特征（排除多维的 RWA/graph_embedding）
    top_df = importance_df[importance_df['feature_group'] != 'graph_embedding'].head(top_k)
    top_features = top_df['feature_group'].values
    
    fig, axes = plt.subplots(1, len(top_features), figsize=(18, 4))
    if len(top_features) == 1: axes = [axes]
    
    for i, feature in enumerate(top_features):
        dep_files = [f for f in os.listdir(nonlinear_dir) if f.endswith(f"{feature}_dependence.csv")]
        
        if not dep_files:
            axes[i].text(0.5, 0.5, f"Missing:\n{feature}", ha='center', va='center')
            continue
        
        data = pd.read_csv(os.path.join(nonlinear_dir, dep_files[0]))
        ax = axes[i]
        
        x = data['feature_value'].values
        y = data['shap_value'].values
        
        # 散点颜色使用你指定的 e67371\bb64dd\498aec
        ax.scatter(x, y, color="#498aec", s=15, alpha=0.5, zorder=1)
        
        # LOWESS 拟合曲线
        if len(x) > 10:
            try:
                filtered_indices = ~np.isnan(x) & ~np.isnan(y)
                smooth = lowess(y[filtered_indices], x[filtered_indices], frac=0.3)
                ax.plot(smooth[:, 0], smooth[:, 1], color="red", linewidth=2.5, zorder=2)
            except:
                pass
        
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_title(name_mapping.get(feature, feature), fontsize=16)
        ax.set_xlabel("Feature Value (Standardized)", fontsize=12)
        if i == 0: ax.set_ylabel("SHAP Value", fontsize=12)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 非线性依赖图已保存: {output_path}")

# --- 执行依赖图绘图 ---
if importance_df is not None:
    dependence_plot_path = os.path.join(xgb_dir, "dependence_top5_standardized.png")
    plot_dependence_top5(importance_df, nonlinear_dir, dependence_plot_path)

print("\n" + "="*50)
print("分析完成！")
print("="*50)