import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ------- 文件路径 -------
# 校准数据在真实节点的距离误差  
calib_path = Path("/home/qwang/projects/leafi/dstree2/result/save_path_QN3/calibration_data.csv")
# 测试数据在knn节点的距离误差
recheck_path = Path("/home/qwang/projects/leafi/dstree2/result/Recheck_error1/1M_check_CP_and_true_Recheck_error_q3k_QN3.csv")
# 预测误差数据
precal_error_path = Path("/home/qwang/projects/leafi/dstree2/result/debug_train_R90_C90_K1_QN2_spline/lgbm_individual_models2/key_points_predictions/pred_errors_recall0.95_cov0.95.txt")
# 错误剪枝记录
wrong_prune_path = Path("/home/qwang/projects/leafi/dstree2/result/save_path_QN3/wrong_pruning_records.csv")

# ------- 读入数据 -------
print("=== 读取数据 ===")
calib = pd.read_csv(calib_path)     # 列名：true_error、node_id 等
recheck = pd.read_csv(recheck_path)   # 列名：prediction_error、filter_id 等
precal = pd.read_csv(precal_error_path)  # 列名：Predicted_Error、Filter_ID 等
wrong_prune = pd.read_csv(wrong_prune_path)  # 列名：node_id、true_error、calib_error 等

print(f"校准数据形状: {calib.shape}")
print(f"测试数据形状: {recheck.shape}")
print(f"预测数据形状: {precal.shape}")
print(f"错误剪枝数据形状: {wrong_prune.shape}")

# 重命名误差列以便更清楚地区分
calib = calib.rename(columns={"true_error": "calib_error"})
recheck = recheck.rename(columns={"prediction_error": "test_error"})
precal = precal.rename(columns={"Predicted_Error": "precal_error", "Filter_ID": "filter_id"})

# 统一列名，确保都有filter_id列
if "filter_id" not in calib.columns:
    calib = calib.rename(columns={"node_id": "filter_id"})

print("\n=== 数据列名检查 ===")
print(f"校准数据列名: {list(calib.columns)}")
print(f"测试数据列名: {list(recheck.columns)}")
print(f"预测数据列名: {list(precal.columns)}")

# 只保留误差大于0的部分
print("\n=== 数据过滤：只保留正误差 ===")
print(f"过滤前 - 校准数据: {len(calib)} 条记录")
print(f"过滤前 - 测试数据: {len(recheck)} 条记录")
print(f"过滤前 - 预测数据: {len(precal)} 条记录")

calib_positive = calib[calib["calib_error"] > 0].copy()
recheck_positive = recheck[recheck["test_error"] > 0].copy()
precal_positive = precal[precal["precal_error"] > 0].copy()

print(f"过滤后 - 校准数据: {len(calib_positive)} 条记录 ({len(calib_positive)/len(calib)*100:.1f}%)")
print(f"过滤后 - 测试数据: {len(recheck_positive)} 条记录 ({len(recheck_positive)/len(recheck)*100:.1f}%)")
print(f"过滤后 - 预测数据: {len(precal_positive)} 条记录 ({len(precal_positive)/len(precal)*100:.1f}%)")

# 更新数据变量，后续分析都使用正误差数据
calib = calib_positive
recheck = recheck_positive
precal = precal_positive

# ------- 按 filter 统计三种误差 -------
print("\n=== 按 filter 计算统计量 ===")

# 校准误差统计
calib_stats = (
    calib.groupby("filter_id")["calib_error"]
    .agg(["count", "mean", "max", "std", "min"])
    .reset_index()
    .rename(columns={"count": "calib_count", "mean": "calib_mean", "max": "calib_max", 
                     "std": "calib_std", "min": "calib_min"})
)

# 测试误差统计
test_stats = (
    recheck.groupby("filter_id")["test_error"]
    .agg(["count", "mean", "max", "std", "min"])
    .reset_index()
    .rename(columns={"count": "test_count", "mean": "test_mean", "max": "test_max",
                     "std": "test_std", "min": "test_min"})
)

# 预测误差统计
precal_stats = (
    precal.groupby("filter_id")["precal_error"]
    .agg(["count", "mean", "max", "std", "min"])
    .reset_index()
    .rename(columns={"count": "precal_count", "mean": "precal_mean", "max": "precal_max",
                     "std": "precal_std", "min": "precal_min"})
)

print(f"校准误差涵盖的filter数量: {len(calib_stats)}")
print(f"测试误差涵盖的filter数量: {len(test_stats)}")
print(f"预测误差涵盖的filter数量: {len(precal_stats)}")

# 合并三个统计表
merged_stats = calib_stats.merge(test_stats, on="filter_id", how="outer")
merged_stats = merged_stats.merge(precal_stats, on="filter_id", how="outer")

# 填充缺失值为0（表示该filter在某种误差类型中没有数据）
error_cols = ["calib_mean", "calib_max", "test_mean", "test_max", "precal_mean", "precal_max"]
merged_stats[error_cols] = merged_stats[error_cols].fillna(0)

print(f"合并后的filter数量: {len(merged_stats)}")
print(f"同时有三种误差数据的filter数量: {len(merged_stats[(merged_stats['calib_mean'] > 0) & (merged_stats['test_mean'] > 0) & (merged_stats['precal_mean'] > 0)])}")

# ------- 保存统计结果 -------
out_dir = Path("/home/qwang/projects/leafi/dstree2/result/three_errors_comparison")
out_dir.mkdir(parents=True, exist_ok=True)

merged_stats.to_csv(out_dir / "three_errors_statistics.csv", index=False)
print(f"\n统计结果已保存到: {out_dir}/three_errors_statistics.csv")

# ------- 可视化：重点关注错误剪枝涉及的filter -------
print("\n=== 创建可视化图表（重点关注错误剪枝filter） ===")

# 只选择有数据的filter进行可视化（至少有一种误差类型有数据）
valid_filters = merged_stats[
    (merged_stats['calib_mean'] > 0) | 
    (merged_stats['test_mean'] > 0) | 
    (merged_stats['precal_mean'] > 0)
].copy()

# 按平均误差大小排序（使用三种误差的平均值）
valid_filters['avg_all_errors'] = (valid_filters['calib_mean'] + 
                                  valid_filters['test_mean'] + 
                                  valid_filters['precal_mean']) / 3

print(f"用于可视化的filter总数量: {len(valid_filters)}")

# 筛选出错误剪枝涉及的filter
wrong_prune_filters = set(wrong_prune['node_id'].unique())
wrong_prune_valid_filters = valid_filters[valid_filters['filter_id'].isin(wrong_prune_filters)].copy()
wrong_prune_valid_filters = wrong_prune_valid_filters.sort_values('avg_all_errors', ascending=True)

print(f"错误剪枝涉及且有误差数据的filter数量: {len(wrong_prune_valid_filters)}")

if len(wrong_prune_valid_filters) == 0:
    print("⚠️  没有找到错误剪枝filter的误差数据，将显示所有filter的随机样本")
    # 如果没有匹配的，则回退到随机选择
    if len(valid_filters) > 50:
        np.random.seed(42)
        sample_filters = valid_filters.sample(n=50, random_state=42).sort_values('avg_all_errors')
        title_suffix = "50 Random Filters (No Wrong-Prune Match)"
    else:
        sample_filters = valid_filters
        title_suffix = f"All {len(sample_filters)} Filters (No Wrong-Prune Match)"
else:
    sample_filters = wrong_prune_valid_filters
    title_suffix = f"{len(sample_filters)} Wrong-Pruning Involved Filters"
    print(f"✅ 成功筛选出错误剪枝涉及的filter: {sorted(sample_filters['filter_id'].tolist())}")

print(f"实际用于可视化的filter数量: {len(sample_filters)}")

# 创建大图表
fig, axes = plt.subplots(2, 1, figsize=(max(16, len(sample_filters) * 0.5), 12))

# 子图1: 平均误差对比
x_pos = np.arange(len(sample_filters))
width = 0.25

ax1 = axes[0]
bars1 = ax1.bar(x_pos - width, sample_filters['calib_mean'], width, 
                label='Calib Error (Mean)', alpha=0.8, color='skyblue')
bars2 = ax1.bar(x_pos, sample_filters['test_mean'], width, 
                label='Test Error (Mean)', alpha=0.8, color='lightcoral')
bars3 = ax1.bar(x_pos + width, sample_filters['precal_mean'], width, 
                label='Precal Error (Mean)', alpha=0.8, color='lightgreen')

ax1.set_xlabel('Filter ID')
ax1.set_ylabel('Mean Error Value')
ax1.set_title(f'Mean Error Comparison - {title_suffix}')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f"{int(fid)}" for fid in sample_filters['filter_id']], 
                    rotation=45, fontsize=max(6, min(10, 200//len(sample_filters))))
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2: 最大误差对比
ax2 = axes[1]
bars4 = ax2.bar(x_pos - width, sample_filters['calib_max'], width, 
                label='Calib Error (Max)', alpha=0.8, color='navy')
bars5 = ax2.bar(x_pos, sample_filters['test_max'], width, 
                label='Test Error (Max)', alpha=0.8, color='darkred')
bars6 = ax2.bar(x_pos + width, sample_filters['precal_max'], width, 
                label='Precal Error (Max)', alpha=0.8, color='darkgreen')

ax2.set_xlabel('Filter ID')
ax2.set_ylabel('Max Error Value')
ax2.set_title(f'Maximum Error Comparison - {title_suffix}')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([f"{int(fid)}" for fid in sample_filters['filter_id']], 
                    rotation=45, fontsize=max(6, min(10, 200//len(sample_filters))))
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "three_errors_comparison_wrong_prune_filters.png", dpi=150, bbox_inches='tight')
plt.show()

# ------- 创建热力图可视化（错误剪枝filter） -------
plt.figure(figsize=(max(12, len(sample_filters) * 0.6), 8))

# 准备热力图数据
heatmap_data = sample_filters[['filter_id', 'calib_mean', 'test_mean', 'precal_mean', 
                               'calib_max', 'test_max', 'precal_max']].copy()
heatmap_data = heatmap_data.set_index('filter_id')
heatmap_data.columns = ['Calib Mean', 'Test Mean', 'Precal Mean', 
                        'Calib Max', 'Test Max', 'Precal Max']

# 创建热力图
sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', 
            cbar_kws={'label': 'Error Value'}, 
            annot_kws={'size': max(6, min(10, 200//len(sample_filters)))})
plt.title(f'Error Heatmap - {title_suffix}')
plt.xlabel('Filter ID')
plt.ylabel('Error Type')
plt.tight_layout()
plt.savefig(out_dir / "three_errors_heatmap_wrong_prune_filters.png", dpi=150, bbox_inches='tight')
plt.show()

# 保存错误剪枝涉及的filter ID列表和统计信息
wrong_prune_filter_ids = sample_filters['filter_id'].tolist()
with open(out_dir / "wrong_prune_filter_analysis.txt", "w") as f:
    f.write("=== 错误剪枝涉及的Filter误差分析 ===\n\n")
    f.write(f"错误剪枝记录总数: {len(wrong_prune)}\n")
    f.write(f"错误剪枝涉及的唯一filter数: {len(wrong_prune_filters)}\n")
    f.write(f"有误差数据的错误剪枝filter数: {len(wrong_prune_valid_filters)}\n\n")
    
    f.write("错误剪枝涉及的filter ID列表:\n")
    f.write(",".join([str(int(fid)) for fid in wrong_prune_filter_ids]) + "\n\n")
    
    if len(wrong_prune_valid_filters) > 0:
        f.write("各filter的三种误差统计:\n")
        f.write("Filter_ID,Calib_Mean,Test_Mean,Precal_Mean,Calib_Max,Test_Max,Precal_Max\n")
        for _, row in sample_filters.iterrows():
            f.write(f"{int(row['filter_id'])},{row['calib_mean']:.4f},{row['test_mean']:.4f},"
                   f"{row['precal_mean']:.4f},{row['calib_max']:.4f},{row['test_max']:.4f},"
                   f"{row['precal_max']:.4f}\n")

print(f"\n错误剪枝filter分析已保存到: {out_dir}/wrong_prune_filter_analysis.txt")

# ------- 统计摘要 -------
print("\n=== 三种误差类型的全局统计摘要 ===")

# 只统计有数据的部分
calib_nonzero = valid_filters[valid_filters['calib_mean'] > 0]['calib_mean']
test_nonzero = valid_filters[valid_filters['test_mean'] > 0]['test_mean']
precal_nonzero = valid_filters[valid_filters['precal_mean'] > 0]['precal_mean']

print("\n平均误差统计:")
print(f"Calib Error  - 有效filter数: {len(calib_nonzero)}, 平均值: {calib_nonzero.mean():.4f}, 标准差: {calib_nonzero.std():.4f}")
print(f"Test Error   - 有效filter数: {len(test_nonzero)}, 平均值: {test_nonzero.mean():.4f}, 标准差: {test_nonzero.std():.4f}")
print(f"Precal Error - 有效filter数: {len(precal_nonzero)}, 平均值: {precal_nonzero.mean():.4f}, 标准差: {precal_nonzero.std():.4f}")

# 最大误差统计
calib_max_nonzero = valid_filters[valid_filters['calib_max'] > 0]['calib_max']
test_max_nonzero = valid_filters[valid_filters['test_max'] > 0]['test_max']
precal_max_nonzero = valid_filters[valid_filters['precal_max'] > 0]['precal_max']

print("\n最大误差统计:")
print(f"Calib Error  - 最大值均值: {calib_max_nonzero.mean():.4f}, 最大值中位数: {calib_max_nonzero.median():.4f}")
print(f"Test Error   - 最大值均值: {test_max_nonzero.mean():.4f}, 最大值中位数: {test_max_nonzero.median():.4f}")
print(f"Precal Error - 最大值均值: {precal_max_nonzero.mean():.4f}, 最大值中位数: {precal_max_nonzero.median():.4f}")

print(f"\n可视化图表已保存到:")
print(f"  - {out_dir}/three_errors_comparison_wrong_prune_filters.png")
print(f"  - {out_dir}/three_errors_heatmap_wrong_prune_filters.png")
print(f"  - {out_dir}/wrong_prune_filter_analysis.txt") 