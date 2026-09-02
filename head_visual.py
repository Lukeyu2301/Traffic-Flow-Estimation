import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ==================== 1. 全局样式设置 ====================
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 23,           # 保持你的字体大小
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'figure.dpi': 250,
})

# 颜色配置：早、中、晚
COLORS = {
    "Morning": "#FDC776",    # 橙色
    "Noon": "#C27CFC",       # 紫色
    "Evening": "#768AFF"     # 蓝色
}

def plot_combined_sensitivity(paths, output_folder):
    # 1. 加载数据
    data = {}
    for period, path in paths.items():
        if os.path.exists(path):
            data[period] = pd.read_csv(path).sort_values('Heads')
        else:
            print(f"❌ 找不到 {period} 的文件: {path}")
            return

    # 2. 绘图参数设置
    heads = data["Morning"]['Heads'].values
    x_pos = np.arange(len(heads))  # 组的位置 [0, 1, 2, 3]
    width = 0.25                    # 单个柱子的宽度
    
    # 3. 创建画布 (1行3列: RMSE, MAE, MAPE)
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = [
        ('RMSE', 'RMSE', axes[0]),
        ('MAE', 'MAE', axes[1]),
        ('MAPE', 'MAPE (%)', axes[2])
    ]

    for metric_key, ylabel, ax in metrics:
        # 分别绘制早、中、晚三组柱子
        # 修改点：edgecolor='none' 去掉黑边, error_kw 设置误差棒颜色
        ax.bar(x_pos - width, data["Morning"][f'{metric_key}_mean'], width, 
               yerr=data["Morning"][f'{metric_key}_std'], 
               color=COLORS["Morning"], label="Morning", 
               edgecolor='none', capsize=5, alpha=1,
               error_kw={'ecolor': "#FDAF39", 'elinewidth': 1.5})
        
        ax.bar(x_pos, data["Noon"][f'{metric_key}_mean'], width, 
               yerr=data["Noon"][f'{metric_key}_std'], 
               color=COLORS["Noon"], label="Noon", 
               edgecolor='none', capsize=5, alpha=1,
               error_kw={'ecolor': "#A333FF", 'elinewidth': 1.5})
        
        ax.bar(x_pos + width, data["Evening"][f'{metric_key}_mean'], width, 
               yerr=data["Evening"][f'{metric_key}_std'], 
               color=COLORS["Evening"], label="Evening", 
               edgecolor='none', capsize=5, alpha=1,
               error_kw={'ecolor': "#3654FC", 'elinewidth': 1.5})

        # 轴标签设置
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{int(h)}" for h in heads])
        # ax.set_ylabel(ylabel, fontsize=23, fontweight='bold', labelpad=10) # 保持标签风格
        
        # 刻度细节
        ax.tick_params(axis='both', which='both', direction='in', pad=8)
        
        # 边框设置
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.5)
        
        # 网格辅助线
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='-.', alpha=0.5, color="#575555")

    # 4. 图例设置 (如果需要显示图例，请取消下面 fig.legend 的注释)
    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=20, frameon=False, bbox_to_anchor=(0.5, 0.98))

    # 5. 控制子图距离
    plt.subplots_adjust(wspace=0.25, top=0.85, bottom=0.15, left=0.08, right=0.96)

    # 6. 保存
    save_path = os.path.join(output_folder, 'heads_sensitivity_all_periods_corrected.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ 汇总柱状图已保存至: {save_path}")

# ==================== 执行脚本 ====================
if __name__ == "__main__":
    file_paths = {
        "Morning": r"D:\KEKE\宣城交通流量预测\results\interp\morning\seed_42\sensitivity_analysis\heads_sensitivity_results_corrected.csv",
        "Noon": r"D:\KEKE\宣城交通流量预测\results\interp\noon\seed_42\sensitivity_analysis\heads_sensitivity_results_corrected.csv",
        "Evening": r"D:\KEKE\宣城交通流量预测\results\interp\evening\seed_42\sensitivity_analysis\heads_sensitivity_results_corrected.csv"
    }
    
    output_dir = r"D:\KEKE\宣城交通流量预测\results\interp\heads_analysis_total"
    os.makedirs(output_dir, exist_ok=True)
    
    plot_combined_sensitivity(file_paths, output_dir)