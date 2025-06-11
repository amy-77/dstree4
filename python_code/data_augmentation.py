#!/usr/bin/env python3
"""
数据增强脚本：在coverage边界添加保守的哨兵值
"""

import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def analyze_data_distribution(data):
    """分析数据分布，确定最佳增强策略"""
    print("数据分布分析:")
    print("=" * 50)
    
    # 按recall分组统计
    recall_groups = data.groupby('recall').agg({
        'coverage': ['min', 'max', 'count'],
        'error': ['min', 'max', 'mean', 'std']
    }).round(6)
    
    print(f"数据总量: {len(data)}")
    print(f"Recall范围: {data['recall'].min():.6f} - {data['recall'].max():.6f}")
    print(f"Coverage范围: {data['coverage'].min():.6f} - {data['coverage'].max():.6f}")
    print(f"Error范围: {data['error'].min():.6f} - {data['error'].max():.6f}")
    
    # 关注高recall区域
    high_recall_data = data[data['recall'] >= 0.99]
    if len(high_recall_data) > 0:
        print(f"\n高Recall区域 (>=0.99) 统计:")
        print(f"数据量: {len(high_recall_data)}")
        print(f"Coverage范围: {high_recall_data['coverage'].min():.6f} - {high_recall_data['coverage'].max():.6f}")
        print(f"Error范围: {high_recall_data['error'].min():.6f} - {high_recall_data['error'].max():.6f}")
    
    return recall_groups


def calculate_sentinel_error(base_error, method='exponential', factor=2.0):
    """计算哨兵误差值
    
    Args:
        base_error: 基础误差值
        method: 增强方法 ('linear', 'exponential', 'quadratic')
        factor: 增强因子
    """
    if method == 'linear':
        return base_error * factor
    elif method == 'exponential':
        return base_error * np.exp(factor - 1)
    elif method == 'quadratic':
        return base_error * (factor ** 2)
    else:
        return base_error * factor


def add_coverage_sentinels(data, method='exponential', factor=2.0, coverage_step=0.001, max_error_at_cov1=10.0):
    """为每个recall添加coverage边界的哨兵值
    
    Args:
        data: 原始数据
        method: 增强方法
        factor: 增强因子
        coverage_step: coverage步长
        max_error_at_cov1: coverage=1.0时的最大error值
    """
    print(f"\n开始数据增强...")
    print(f"增强方法: {method}, 因子: {factor}")
    print(f"Coverage步长: {coverage_step}")
    print(f"Coverage=1.0时的最大error: {max_error_at_cov1}")
    
    # 解释指数增强的数学原理
    if method == 'exponential':
        multiplier = np.exp(factor - 1)
        print(f"指数增强计算公式: new_error = base_error * exp({factor} - 1) = base_error * {multiplier:.3f}")
        print(f"这意味着基础误差会被乘以 {multiplier:.3f} 倍")
    
    augmented_data = data.copy()
    new_points = []
    
    print(f"\n详细增强过程:")
    print("-" * 80)
    
    # 按recall分组处理
    for recall_idx, (recall, group) in enumerate(data.groupby('recall')):
        print(f"\nRecall组 {recall_idx + 1}: {recall:.6f}")
        
        # 检查是否存在coverage=1.0的情况
        cov_1_exists = (group['coverage'] >= 1.0).any()
        
        if cov_1_exists:
            print(f"  检测到coverage=1.0的数据点，将调整为coverage=0.9995")
            # 在augmented_data中调整coverage=1.0为0.9995
            mask = (augmented_data['recall'] == recall) & (augmented_data['coverage'] >= 1.0)
            augmented_data.loc[mask, 'coverage'] = 0.9995
            
            # 更新group数据
            group = augmented_data[augmented_data['recall'] == recall]
        
        max_coverage = group['coverage'].max()
        max_error = group[group['coverage'] == max_coverage]['error'].iloc[0]
        
        print(f"  原始最大coverage: {max_coverage:.6f}, 对应error: {max_error:.6f}")
        
        # 确定coverage=1.0时应该使用的error值：总是使用当前最大coverage对应error的3倍
        max_error_for_cov1 = max_error * 3.0
        
        print(f"  添加coverage=1.0的极值点，error={max_error_for_cov1:.6f}")
        
        # 总是为每个recall组添加coverage=1.0的数据点
        new_point = {
            'recall': recall,
            'coverage': 1.0,
            'error': max_error_for_cov1,
            'is_test': 0  # 新增数据都作为训练集
        }
        if 'filter_id' in data.columns:
            new_point['filter_id'] = group['filter_id'].iloc[0]
        new_points.append(new_point)
        
        # 对从最大coverage到1.0之间的区间进行插值，步长0.01
        interpolation_step = 0.01
        
        # 计算需要插值的coverage点
        coverage_start = max_coverage + interpolation_step
        
        # 向上取整到最近的0.01倍数
        coverage_start = np.ceil(coverage_start / interpolation_step) * interpolation_step
        
        interpolation_coverages = []
        current_cov = coverage_start
        while current_cov < 1.0:
            interpolation_coverages.append(round(current_cov, 2))  # 保留2位小数避免浮点误差
            current_cov += interpolation_step
        
        if len(interpolation_coverages) > 0:
            print(f"  将对coverage={max_coverage:.6f}到1.0之间进行插值，步长0.01")
            print(f"  插值范围: {interpolation_coverages[0]:.2f} 到 {interpolation_coverages[-1]:.2f}，共{len(interpolation_coverages)}个点")
            
            # 线性插值计算每个coverage点的error值
            for interp_coverage in interpolation_coverages:
                # 线性插值公式
                ratio = (interp_coverage - max_coverage) / (1.0 - max_coverage)
                interp_error = max_error + (max_error_for_cov1 - max_error) * ratio
                
                print(f"    插值点: coverage={interp_coverage:.2f} -> error={interp_error:.6f}")
                
                # 添加插值点
                new_point = {
                    'recall': recall,
                    'coverage': interp_coverage,
                    'error': interp_error,
                    'is_test': 0  # 新增数据都作为训练集
                }
                
                if 'filter_id' in data.columns:
                    new_point['filter_id'] = group['filter_id'].iloc[0]
                
                new_points.append(new_point)
        else:
            print(f"  跳过插值 (最大coverage已接近1.0)")
        
        # 跳过原有的中间哨兵点生成逻辑，因为我们已经用插值替代了
        continue
    
    print("-" * 80)
    
    # 添加新点到数据集
    if new_points:
        new_df = pd.DataFrame(new_points)
        augmented_data = pd.concat([augmented_data, new_df], ignore_index=True)
        print(f"\n✓ 总共添加了 {len(new_points)} 个哨兵数据点")
        print(f"  (全部为训练样本)")
        
        # 统计增强效果
        orig_max_error = data['error'].max()
        new_max_error = augmented_data['error'].max()
        print(f"  原始最大error: {orig_max_error:.6f}")
        print(f"  增强后最大error: {new_max_error:.6f}")
        print(f"  最大error增长: {new_max_error/orig_max_error:.2f}x")
    else:
        print(f"\n⚠️  没有添加任何哨兵点 (所有recall组都已达到coverage上限)")
    
    # 四舍五入error列到6位小数，避免浮点精度问题
    augmented_data['error'] = augmented_data['error'].round(6)
    
    # 按recall升序，然后按coverage升序排序
    augmented_data = augmented_data.sort_values(['recall', 'coverage']).reset_index(drop=True)
    
    return augmented_data


def add_extreme_sentinels(data, extreme_coverages=[0.999, 0.9999]):
    """为所有recall添加极高coverage的哨兵值"""
    print(f"\n添加极值哨兵点...")
    
    augmented_data = data.copy()
    new_points = []
    
    for recall, group in data.groupby('recall'):
        max_error = group['error'].max()
        
        for extreme_cov in extreme_coverages:
            # 使用指数增长计算极值误差
            if extreme_cov == 0.999:
                extreme_error = max_error * 3.0  # 3倍增长
            elif extreme_cov == 0.9999:
                extreme_error = max_error * 5.0  # 5倍增长
            else:
                extreme_error = max_error * 2.0
            
            # 添加训练集样本
            new_point = {
                'recall': recall,
                'coverage': extreme_cov,
                'error': extreme_error,
                'is_test': 0  # 新增数据都作为训练集
            }
            
            if 'filter_id' in data.columns:
                new_point['filter_id'] = group['filter_id'].iloc[0]
            
            new_points.append(new_point)
    
    if new_points:
        new_df = pd.DataFrame(new_points)
        augmented_data = pd.concat([augmented_data, new_df], ignore_index=True)
        print(f"添加了 {len(new_points)} 个极值哨兵点")
    
    return augmented_data


def visualize_augmentation(original_data, augmented_data, output_dir):
    """可视化数据增强效果"""
    plt.figure(figsize=(15, 10))
    # 原始数据
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(original_data['coverage'], original_data['recall'], 
                         c=original_data['error'], cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Error')
    plt.xlabel('Coverage')
    plt.ylabel('Recall')
    plt.title('Original Data Distribution')
    plt.grid(True, alpha=0.3)
    # 增强后数据
    plt.subplot(2, 2, 2)
    scatter = plt.scatter(augmented_data['coverage'], augmented_data['recall'], 
                         c=augmented_data['error'], cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Error')
    plt.xlabel('Coverage')
    plt.ylabel('Recall')
    plt.title('Augmented Data Distribution')
    plt.grid(True, alpha=0.3)
    
    # 高recall区域对比
    high_recall_orig = original_data[original_data['recall'] >= 0.99]
    high_recall_aug = augmented_data[augmented_data['recall'] >= 0.99]
    
    plt.subplot(2, 2, 3)
    if len(high_recall_orig) > 0:
        plt.scatter(high_recall_orig['coverage'], high_recall_orig['error'], 
                   alpha=0.7, s=30, label='Original Data', color='blue')
    new_points = augmented_data[~augmented_data.index.isin(original_data.index)]
    high_recall_new = new_points[new_points['recall'] >= 0.99]
    if len(high_recall_new) > 0:
        plt.scatter(high_recall_new['coverage'], high_recall_new['error'], 
                   alpha=0.7, s=30, label='New Sentinels', color='red', marker='^')
    plt.xlabel('Coverage')
    plt.ylabel('Error')
    plt.title('High Recall Region (>=0.99) Augmentation Effect')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Error分布对比
    plt.subplot(2, 2, 4)
    plt.hist(original_data['error'], bins=50, alpha=0.5, label='Original Data', color='blue')
    plt.hist(augmented_data['error'], bins=50, alpha=0.5, label='Augmented Data', color='red')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # 保存图片
    plot_path = os.path.join(output_dir, 'data_augmentation_visualization.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"可视化图已保存到: {plot_path}")
    plt.show()




def main():
    parser = argparse.ArgumentParser(description='数据增强：添加coverage边界哨兵值')
    parser.add_argument('--input', type=str, required=True, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, help='输出CSV文件路径')
    parser.add_argument('--method', type=str, choices=['linear', 'exponential', 'quadratic'], 
                       default='exponential', help='增强方法')
    parser.add_argument('--factor', type=float, default=2.0, help='增强因子')
    parser.add_argument('--coverage_step', type=float, default=0.001, help='Coverage步长')
    parser.add_argument('--max_error_at_cov1', type=float, default=10.0, help='Coverage=1.0时的最大error值')
    parser.add_argument('--add_extremes', action='store_true', help='添加极值哨兵点')
    parser.add_argument('--visualize', action='store_true', help='生成可视化图')
    
    args = parser.parse_args()
    
    # 加载数据
    if not os.path.exists(args.input):
        print(f"输入文件不存在: {args.input}")
        return 1
    
    try:
        data = pd.read_csv(args.input)
        print(f"✓ 数据已加载: {args.input}")
        print(f"  原始数据量: {len(data)}")
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return 1
    
    # 分析数据分布
    analyze_data_distribution(data)
    
    # 数据增强
    augmented_data = add_coverage_sentinels(data, args.method, args.factor, args.coverage_step, args.max_error_at_cov1)
    
    if args.add_extremes:
        augmented_data = add_extreme_sentinels(augmented_data)
    
    # 输出统计
    print(f"\n数据增强完成!")
    print(f"原始数据量: {len(data)}")
    print(f"增强后数据量: {len(augmented_data)}")
    print(f"新增数据量: {len(augmented_data) - len(data)}")
    
    # 保存结果
    if args.output:
        output_path = args.output
    else:
        input_dir = os.path.dirname(args.input)
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = os.path.join(input_dir, f"{input_name}_augmented.csv")
    
    # 四舍五入error列到6位小数，避免浮点精度问题
    augmented_data['error'] = augmented_data['error'].round(6)
    
    # 按recall升序，然后按coverage升序排序
    augmented_data = augmented_data.sort_values(['recall', 'coverage']).reset_index(drop=True)
    
    augmented_data.to_csv(output_path, index=False)
    print(f"✓ 增强数据已保存到: {output_path}")
    
    # 可视化
    if args.visualize:
        output_dir = os.path.dirname(output_path)
        visualize_augmentation(data, augmented_data, output_dir)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main()) 