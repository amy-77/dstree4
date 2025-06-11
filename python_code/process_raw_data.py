#!/usr/bin/env python3
"""
处理原始数据文件，对相同的recall和coverage组合取最大的error值
"""

import pandas as pd
import numpy as np
import os
import sys


def process_raw_data(input_path, output_path, precision=6):
    """
    处理原始数据，对相同的recall和coverage组合，找到最大误差所在的列的最大误差
    
    Args:
        input_path: 输入的原始数据文件路径
        output_path: 输出的处理后数据文件路径
        precision: 四舍五入的精度位数
    """
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在 - {input_path}")
        return 1
    
    # 读取原始数据
    try:
        print(f"正在读取原始数据: {input_path}")
        data = pd.read_csv(input_path)
        print(f"✓ 原始数据已加载，共 {len(data)} 行")
        
        # 显示数据结构
        print(f"数据列: {list(data.columns)}")
        print(f"数据前5行:")
        print(data.head())
        
    except Exception as e:
        print(f"错误: 无法读取数据文件 - {e}")
        return 1
    
    # 处理多个recall/cov/error组合
    all_records = []
    
    # 查找所有的recall/cov/error列组合
    recall_cols = [col for col in data.columns if col.startswith('recall')]
    cov_cols = [col for col in data.columns if col.startswith('cov')]
    error_cols = [col for col in data.columns if col.startswith('actual error')]
    
    print(f"\n发现的列:")
    print(f"  Recall列: {len(recall_cols)} 个")
    print(f"  Coverage列: {len(cov_cols)} 个") 
    print(f"  Error列: {len(error_cols)} 个")
    
    # 确定列组的数量，应该是最小的列数
    num_groups = min(len(recall_cols), len(cov_cols), len(error_cols))
    print(f"  有效数据组数: {num_groups}")
    
    # 提取所有数据点，包含列信息
    for i, row in data.iterrows():
        # 处理所有找到的error列
        for j, error_col in enumerate(error_cols):
            # 根据error列的索引找对应的recall和cov列
            if j < len(recall_cols) and j < len(cov_cols):
                recall_col = recall_cols[j]
                cov_col = cov_cols[j]
                
                recall_val = row[recall_col]
                cov_val = row[cov_col]
                error_val = row[error_col]
                
                if pd.notna(recall_val) and pd.notna(cov_val) and pd.notna(error_val):
                    all_records.append({
                        'recall': recall_val,
                        'coverage': cov_val,
                        'error': error_val,
                        'row_index': i,
                        'col_group': j,  # 记录来自哪个列组
                        'error_col': error_col  # 记录具体的error列名
                    })
    
    # 转换为DataFrame
    if not all_records:
        print("错误: 没有找到有效的数据记录")
        return 1
    
    expanded_data = pd.DataFrame(all_records)
    print(f"\n✓ 展开后的数据: {len(expanded_data)} 条记录")
    
    # 对recall和coverage进行四舍五入，避免浮点数精度问题
    print(f"\n正在对recall和coverage进行四舍五入 (精度: {precision}位小数)")
    expanded_data['recall_rounded'] = expanded_data['recall'].round(precision)
    expanded_data['coverage_rounded'] = expanded_data['coverage'].round(precision)
    
    # 显示原始数据统计
    print(f"\n原始数据统计:")
    print(f"  总样本数: {len(expanded_data)}")
    print(f"  Recall范围: [{expanded_data['recall'].min():.6f}, {expanded_data['recall'].max():.6f}]")
    print(f"  Coverage范围: [{expanded_data['coverage'].min():.6f}, {expanded_data['coverage'].max():.6f}]")
    print(f"  Error范围: [{expanded_data['error'].min():.6f}, {expanded_data['error'].max():.6f}]")
    print(f"  唯一的(recall, coverage)组合数: {len(expanded_data[['recall_rounded', 'coverage_rounded']].drop_duplicates())}")
    
    # 新的处理逻辑：对相同(recall, coverage)组合，取包含该组合的所有列组的全局最大误差中的最大值
    print(f"\n正在处理数据: 对相同(recall, coverage)组合，取包含该组合的所有列组的全局最大误差中的最大值...")

    # 预先计算每个列组在整个数据集中的全局最大误差
    global_max_error_by_colgroup = {}
    for idx, err_col in enumerate(error_cols):
        col_errors = data[err_col].dropna()
        if len(col_errors) > 0:
            global_max_error_by_colgroup[idx] = col_errors.max()
        else:
            global_max_error_by_colgroup[idx] = np.nan

    processed_records = []

    # 按照rounded的recall和coverage分组
    groups = expanded_data.groupby(['recall_rounded', 'coverage_rounded'])

    for (recall_rounded, coverage_rounded), group in groups:
        # 该(recall, coverage)组合出现过的列组索引
        involved_col_groups = group['col_group'].unique()

        # 这些列组对应的全局最大误差
        candidate_errors = [global_max_error_by_colgroup.get(idx, np.nan) for idx in involved_col_groups]
        candidate_errors = [err for err in candidate_errors if pd.notna(err)]

        if candidate_errors:
            final_error = max(candidate_errors)
            chosen_col_group = involved_col_groups[np.argmax(candidate_errors)]
        else:
            final_error = 0.0
            chosen_col_group = None

        first_record = group.iloc[0]
        print(
            f"  (recall={recall_rounded:.6f}, coverage={coverage_rounded:.6f}) -> 列组{list(involved_col_groups)}; "
            f"选择列组 {chosen_col_group} 的全局最大误差 {final_error:.6f}"
        )

        processed_records.append({
            'recall': first_record['recall'],
            'coverage': first_record['coverage'],
            'error': final_error
        })

    # 转换为DataFrame
    processed_data = pd.DataFrame(processed_records)
    
    # 添加is_test列
    # 简单策略：80%训练，20%测试
    n_samples = len(processed_data)
    n_test = int(n_samples * 0.2)
    
    # 随机选择测试样本
    np.random.seed(42)  # 固定随机种子保证可重复性
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    
    processed_data['is_test'] = 0
    processed_data.loc[test_indices, 'is_test'] = 1
    
    print(f"✓ 自动添加is_test列: {n_samples - n_test} 训练样本, {n_test} 测试样本")
    
    # 显示处理后数据统计
    print(f"\n处理后数据统计:")
    print(f"  总样本数: {len(processed_data)}")
    print(f"  训练样本数: {len(processed_data[processed_data['is_test'] == 0])}")
    print(f"  测试样本数: {len(processed_data[processed_data['is_test'] == 1])}")
    print(f"  Recall范围: [{processed_data['recall'].min():.6f}, {processed_data['recall'].max():.6f}]")
    print(f"  Coverage范围: [{processed_data['coverage'].min():.6f}, {processed_data['coverage'].max():.6f}]")
    print(f"  Error范围: [{processed_data['error'].min():.6f}, {processed_data['error'].max():.6f}]")
    
    # 按recall和coverage排序
    processed_data = processed_data.sort_values(['recall', 'coverage']).reset_index(drop=True)
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存处理后的数据
    try:
        processed_data.to_csv(output_path, index=False)
        print(f"\n✓ 处理后的数据已保存到: {output_path}")
        
        # 显示保存的数据前几行
        print(f"\n保存的数据前5行:")
        print(processed_data.head())
        
        # 显示一些统计信息
        print(f"\n数据质量检查:")
        duplicate_pairs = processed_data.groupby(['recall', 'coverage']).size()
        max_duplicates = duplicate_pairs.max()
        if max_duplicates > 1:
            print(f"⚠ 发现重复的(recall, coverage)组合，最大重复次数: {max_duplicates}")
            duplicated_coords = duplicate_pairs[duplicate_pairs > 1]
            print(f"重复的组合数量: {len(duplicated_coords)}")
        else:
            print(f"✓ 所有(recall, coverage)组合都是唯一的")
        
        return 0
        
    except Exception as e:
        print(f"错误: 无法保存数据文件 - {e}")
        return 1


def main():
    """主函数"""
    
    # 默认路径
    input_path = "/home/qwang/projects/leafi/dstree2/result/test_error/filter_9_raw_data.csv"
    output_path = "/home/qwang/projects/leafi/dstree2/result/test_error/filter_9_processed_max_error.csv"
    
    print("="*80)
    print("原始数据处理工具 - 对相同(recall, coverage)组合取最大误差所在列的最大误差")
    print("="*80)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print("-"*80)
    
    result = process_raw_data(input_path, output_path)
    
    if result == 0:
        print("\n" + "="*80)
        print("✓ 数据处理完成！")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("✗ 数据处理失败！")
        print("="*80)
    
    return result


if __name__ == "__main__":
    sys.exit(main()) 