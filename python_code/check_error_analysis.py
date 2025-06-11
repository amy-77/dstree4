import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置更好的可视化风格
plt.style.use('ggplot')
if 'sns' in globals():
    sns.set_style("whitegrid")

# 文件路径
calibration_file = "/home/qwang/projects/leafi/dstree2/result/debug_train_R90_C90_K1_QN2_spline/lgbm_individual_models/key_points_predictions/pred_errors_recall0.95_cov0.95.txt"
# real_query_file = "/home/qwang/projects/leafi/dstree2/result/Recheck_error/Recheck_error_q10k.csv"
real_query_file = "/home/qwang/projects/leafi/dstree2/result/save_path_QN11/wrong_pruning_records.csv"
# 读取校准集误差数据
calib_df = pd.read_csv(calibration_file)
print(f"校准集数据形状: {calib_df.shape}")

# 读取真实查询误差数据
query_df = pd.read_csv(real_query_file)
print(f"真实查询数据形状: {query_df.shape}")

# 确保列名匹配
if 'Predicted_Error' in calib_df.columns:
    calib_df['calibration_error'] = calib_df['Predicted_Error']
else:
    calib_df['calibration_error'] = calib_df['Predicted_Error']

if 'Filter_ID' in calib_df.columns:
    calib_df['node_id'] = calib_df['Filter_ID']

# 过滤只保留大于0的真实误差数据
query_df_positive = query_df[query_df['true_error'] > 0].copy()
print(f"过滤后保留正误差的查询数量: {len(query_df_positive)} ({len(query_df_positive)/len(query_df)*100:.2f}%)")

# 同样对校准数据也过滤只保留正误差
print(f"校准数据过滤前: {len(calib_df)} 条记录")
calib_df_positive = calib_df[calib_df['calibration_error'] > 0].copy()
print(f"校准数据过滤后: {len(calib_df_positive)} 条记录 ({len(calib_df_positive)/len(calib_df)*100:.2f}%)")

# 更新变量
calib_df = calib_df_positive
query_df = query_df_positive

# 计算每个node的正误差平均值和样本数量
node_stats = query_df.groupby('node_id')['true_error'].agg(['mean', 'count', 'std']).reset_index()
node_stats = node_stats.rename(columns={'mean': 'avg_real_error'})

# 只保留样本数足够的node (至少5个样本)
node_stats = node_stats[node_stats['count'] >= 5]
print(f"至少有5个正误差样本的node数量: {len(node_stats)}")

# 合并校准误差数据
merged_df = pd.merge(node_stats, calib_df[['node_id', 'calibration_error']], on='node_id', how='inner')
print(f"合并后的数据形状: {merged_df.shape}")

# 计算误差差异 (真实误差 - 校准误差)
merged_df['error_difference'] = merged_df['avg_real_error'] - merged_df['calibration_error']

# 创建一个大图表
plt.figure(figsize=(18, 12))

# 设置字体大小
plt.rcParams.update({'font.size': 12})

# 1. 散点图 - 比较校准误差和平均真实误差
plt.subplot(2, 2, 1)
scatter = plt.scatter(merged_df['calibration_error'], merged_df['avg_real_error'], 
                     alpha=0.7, s=merged_df['count']/2, c=merged_df['error_difference'], cmap='coolwarm')

# 添加对角线
min_val = min(merged_df['calibration_error'].min(), merged_df['avg_real_error'].min())
max_val = max(merged_df['calibration_error'].max(), merged_df['avg_real_error'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Equal Line')

# 添加colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('Difference (Real - Calib)')

# 添加node ID标签
for _, row in merged_df.iterrows():
    plt.annotate(f"{int(row['node_id'])}", 
                 (row['calibration_error'], row['avg_real_error']),
                 xytext=(5, 0), textcoords='offset points', fontsize=8)

plt.title('Calibration Error vs. Average Real Error (≥0)')
plt.xlabel('Calibration Error')
plt.ylabel('Average Real Error (≥0)')
plt.grid(True, alpha=0.3)
plt.legend()

# 2. 误差差异条形图
plt.subplot(2, 2, 2)
# 按差异排序
merged_df_sorted = merged_df.sort_values('error_difference')
# 设置颜色 (蓝色=真实误差小于校准误差，红色=真实误差大于校准误差)
colors = ['blue' if x <= 0 else 'red' for x in merged_df_sorted['error_difference']]
bars = plt.bar(range(len(merged_df_sorted)), merged_df_sorted['error_difference'], color=colors)

# 添加标签
for i, (_, row) in enumerate(merged_df_sorted.iterrows()):
    if i % max(1, len(merged_df_sorted) // 20) == 0:  # 只显示部分标签防止拥挤
        plt.text(i, row['error_difference'], f"{int(row['node_id'])}", 
                 ha='center', va='bottom' if row['error_difference'] > 0 else 'top', fontsize=8)

plt.axhline(y=0, color='black', linestyle='-')
plt.title('Difference Between Real Error and Calibration Error (Real - Calib)')
plt.xlabel('Nodes (sorted by difference)')
plt.ylabel('Error Difference')
plt.grid(True, alpha=0.3)

# 3. 样本数量条形图
plt.subplot(2, 2, 3)
merged_df_count_sorted = merged_df.sort_values('count', ascending=False)
plt.bar(range(len(merged_df_count_sorted)), merged_df_count_sorted['count'], color='green', alpha=0.6)
plt.title('Sample Count by Node')
plt.xlabel('Nodes (sorted by sample count)')
plt.ylabel('Number of Samples')
plt.grid(True, alpha=0.3)

# 4. 校准误差和真实误差的分布对比
plt.subplot(2, 2, 4)
plt.hist(merged_df['calibration_error'], bins=20, alpha=0.5, label='Calibration Error')
plt.hist(merged_df['avg_real_error'], bins=20, alpha=0.5, label='Avg Real Error (≥0)')
plt.axvline(merged_df['calibration_error'].mean(), color='blue', linestyle='--', 
            label=f'Calib Mean: {merged_df["calibration_error"].mean():.3f}')
plt.axvline(merged_df['avg_real_error'].mean(), color='orange', linestyle='--', 
            label=f'Real Mean: {merged_df["avg_real_error"].mean():.3f}')
plt.title('Distribution of Errors')
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/qwang/projects/leafi/dstree2/result/save_path_QN11/pruning_error_comparison_summary.png', dpi=300)
plt.close()

# 打印真实误差大于校准误差的node
error_greater_nodes = merged_df[merged_df['error_difference'] > 0].sort_values('error_difference', ascending=False)

print("\n真实误差大于校准误差的node:")
print(f"总共有 {len(error_greater_nodes)} 个node的真实误差大于校准误差")
print("\n真实误差大于校准误差的node详情（按差异从大到小排序）:")
print("=" * 80)
print(f"{'Node ID':<10} {'样本数':<10} {'真实误差':<15} {'校准误差':<15} {'差异':<15}")
print("-" * 80)

for _, row in error_greater_nodes.iterrows():
    print(f"{int(row['node_id']):<10} {int(row['count']):<10} {row['avg_real_error']:<15.4f} {row['calibration_error']:<15.4f} {row['error_difference']:<15.4f}")

print("=" * 80)

# 计算总体统计数据
print("\n总体统计:")
print(f"所有node的平均校准误差: {merged_df['calibration_error'].mean():.4f}")
print(f"所有node的平均真实误差(≥0): {merged_df['avg_real_error'].mean():.4f}")
print(f"误差差异的平均值: {merged_df['error_difference'].mean():.4f}")
print(f"误差差异的标准差: {merged_df['error_difference'].std():.4f}")

# 保存结果到CSV
merged_df.to_csv('/home/qwang/projects/leafi/dstree2/result/save_path_QN11/node_pruning_error_comparison.csv', index=False)



