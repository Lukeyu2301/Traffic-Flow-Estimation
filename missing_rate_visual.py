import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 全局字体与样式设置 ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 23,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'figure.dpi': 250,
})

MARKER_SIZE = 8
LINE_WIDTH = 2.2
FILL_ALPHA = 0.15  # 误差区间透明度

# 颜色与标记配置
MODEL_CONFIG = {
    "GAT-XGBoost": {"color": "#EC4D9F", "marker": "s", "label": "Proposed (GAT-XGBoost)"},
    "XGBoost":     {"color": "#36D892", "marker": "o", "label": "Baseline (XGBoost)"},
    "GAT-MLP":     {"color": "#206CDF", "marker": "^", "label": "Baseline (GAT-MLP)"}
}

def plot_two_model_comparison(df, output_dir, models_to_plot, save_name):
    """绘制两个模型的 1x3 对比图"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = [
        ('RMSE', 'RMSE', axes[0]),
        ('MAE', 'MAE', axes[1]),
        ('MAPE', 'MAPE (%)', axes[2]),
    ]
    
    x_raw = sorted(df['Missing_Rate'].unique())
    x_percent = [v * 100 for v in x_raw]
    
    for metric_key, ylabel, ax in metrics:
        for model_name in models_to_plot:
            
            config = MODEL_CONFIG[model_name]
            sub_df = df[df['Model'] == model_name].sort_values('Missing_Rate')
            if sub_df.empty:
                continue
            
            y = sub_df[f'{metric_key}_mean']
            std = sub_df[f'{metric_key}_std']
            
            ax.plot(
                x_percent, y,
                marker=config['marker'],
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                color=config['color'],
                label=config['label'],
                zorder=3
            )
            
            ax.fill_between(
                x_percent,
                y - std,
                y + std,
                alpha=FILL_ALPHA,
                color=config['color'],
                edgecolor='none',
                zorder=2
            )
        
        ax.set_xticks(x_percent)
        ax.set_xticklabels([f'{int(v)}' for v in x_percent])
        
        # 统一边框样式
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1.3)

    # 图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False)

    plt.subplots_adjust(wspace=0.25, left=0.08, right=0.95, bottom=0.15, top=0.82)
    
    save_path = os.path.join(output_dir, save_name + '.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"✅ 已保存对比图: {save_path}")
    
    plt.close()


def main():
    # ========== 路径配置 ==========
    input_csv = r"D:\KEKE\宣城交通流量预测\results\interp\morning\seed_42\sparsity_nested_optimized_results_corrected.csv"
    output_dir = r"D:\KEKE\宣城交通流量预测\results\interp\morning\seed_42\figures"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_csv):
        print(f"❌ 找不到文件: {input_csv}")
        return
        
    df = pd.read_csv(input_csv)
    print("成功加载实验数据，包含模型:", df['Model'].unique())
    
    print("\n正在生成双模型对比图...")
    
    # ✅ 图1：XGBoost vs GAT-XGBoost
    plot_two_model_comparison(
        df,
        output_dir,
        models_to_plot=["XGBoost", "GAT-XGBoost"],
        save_name="comparison_XGBoost_vs_GAT-XGBoost_0824"
    )
    
    # ✅ 图2：GAT-MLP vs GAT-XGBoost
    plot_two_model_comparison(
        df,
        output_dir,
        models_to_plot=["GAT-MLP", "GAT-XGBoost"],
        save_name="comparison_GAT-MLP_vs_GAT-XGBoost_0824"
    )
    
    print("\n" + "="*50)
    print(f"可视化完成！结果保存在: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()