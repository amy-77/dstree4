#!/usr/bin/env python3
"""
增强特征模型预测脚本
专门用于对使用特征工程训练的LightGBM模型进行预测
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import sys


def enhance_features_for_prediction(data):
    """
    为预测数据进行特征工程 - 与训练时完全一致
    
    Args:
        data: 包含 recall 和 coverage 列的 DataFrame
    
    Returns:
        增强后的 DataFrame，包含所有30个特征
    """
    if isinstance(data, np.ndarray):
        # 如果输入是numpy数组，转换为DataFrame
        df = pd.DataFrame(data, columns=['recall', 'coverage'])
    else:
        df = data.copy()
    
    # 确保有recall和coverage列
    if 'recall' not in df.columns or 'coverage' not in df.columns:
        raise ValueError("数据必须包含 'recall' 和 'coverage' 列")
    
    print("正在进行特征工程...")
    
    # 1. 交互特征
    df['recall_coverage_product'] = df['recall'] * df['coverage']
    df['recall_coverage_ratio'] = df['recall'] / (df['coverage'] + 1e-10)
    df['coverage_recall_ratio'] = df['coverage'] / (df['recall'] + 1e-10)
    
    # 2. 多项式特征
    df['recall_squared'] = df['recall'] ** 2
    df['coverage_squared'] = df['coverage'] ** 2
    df['recall_cubed'] = df['recall'] ** 3
    df['coverage_cubed'] = df['coverage'] ** 3
    
    # 3. 对数特征（提高极值区域敏感性）
    df['log_recall'] = np.log(df['recall'] + 1e-10)
    df['log_coverage'] = np.log(df['coverage'] + 1e-10)
    df['log_product'] = df['log_recall'] + df['log_coverage']
    
    # 4. 距离特征
    df['distance_to_perfect'] = np.sqrt((1-df['recall'])**2 + (1-df['coverage'])**2)
    df['distance_to_origin'] = np.sqrt(df['recall']**2 + df['coverage']**2)
    df['manhattan_distance_to_perfect'] = (1-df['recall']) + (1-df['coverage'])
    
    # 5. 三角函数特征（捕捉周期性模式）
    df['recall_sin'] = np.sin(df['recall'] * np.pi)
    df['coverage_sin'] = np.sin(df['coverage'] * np.pi)
    df['recall_cos'] = np.cos(df['recall'] * np.pi)
    df['coverage_cos'] = np.cos(df['coverage'] * np.pi)
    
    # 6. 指数特征
    df['exp_recall'] = np.exp(-df['recall'])
    df['exp_coverage'] = np.exp(-df['coverage'])
    
    # 7. 阈值特征（捕捉不同区域的特性）
    df['high_recall'] = (df['recall'] > 0.95).astype(int)
    df['high_coverage'] = (df['coverage'] > 0.95).astype(int)
    df['both_high'] = df['high_recall'] * df['high_coverage']
    
    # 8. 差值特征
    df['recall_coverage_diff'] = df['recall'] - df['coverage']
    df['abs_recall_coverage_diff'] = np.abs(df['recall_coverage_diff'])
    
    # 9. 复合特征
    df['harmonic_mean'] = 2 * df['recall'] * df['coverage'] / (df['recall'] + df['coverage'] + 1e-10)
    df['geometric_mean'] = np.sqrt(df['recall'] * df['coverage'])
    
    # 10. 分段线性特征（模拟分段函数）
    df['recall_high_region'] = np.maximum(0, df['recall'] - 0.9)
    df['coverage_high_region'] = np.maximum(0, df['coverage'] - 0.9)
    
    print(f"特征工程完成，从 2 个基础特征扩展到 {len(df.columns)} 个特征")
    
    return df


def get_feature_columns_for_prediction():
    """
    获取用于预测的特征列名，与训练时完全一致
    
    Returns:
        特征列名列表（30个特征）
    """
    # 基础特征
    feature_cols = ['recall', 'coverage']
    
    # 增强特征（与训练脚本完全一致）
    enhanced_features = [
        'recall_coverage_product', 'recall_coverage_ratio', 'coverage_recall_ratio',
        'recall_squared', 'coverage_squared', 'recall_cubed', 'coverage_cubed',
        'log_recall', 'log_coverage', 'log_product',
        'distance_to_perfect', 'distance_to_origin', 'manhattan_distance_to_perfect',
        'recall_sin', 'coverage_sin', 'recall_cos', 'coverage_cos',
        'exp_recall', 'exp_coverage',
        'high_recall', 'high_coverage', 'both_high',
        'recall_coverage_diff', 'abs_recall_coverage_diff',
        'harmonic_mean', 'geometric_mean',
        'recall_high_region', 'coverage_high_region'
    ]
    
    return feature_cols + enhanced_features


def load_original_data(data_path):
    """加载原始训练数据用于对比"""
    if not os.path.exists(data_path):
        print(f"原始数据文件不存在: {data_path}")
        return None
    
    try:
        data = pd.read_csv(data_path)
        print(f"✓ 原始数据已加载: {data_path}")
        print(f"  数据点数量: {len(data)}")
        return data
    except Exception as e:
        print(f"✗ 加载原始数据失败: {e}")
        return None


def find_actual_error_with_info(data, recall, coverage, tolerance=0.001):
    """在原始数据中查找对应的实际error值，并返回匹配类型信息"""
    if data is None:
        return np.nan, "NoData", np.nan, np.nan
    
    # 首先尝试精确匹配（允许小的误差容忍度）
    matches = data[
        (abs(data['recall'] - recall) <= tolerance) & 
        (abs(data['coverage'] - coverage) <= tolerance)
    ]
    
    if len(matches) > 0:
        # 如果有多个匹配点，返回平均值
        return matches['error'].mean(), "Exact", recall, coverage
    
    # 如果没有精确匹配，找距离最近的点
    # 计算欧几里得距离
    distances = np.sqrt((data['recall'] - recall)**2 + (data['coverage'] - coverage)**2)
    min_distance = distances.min()
    
    # 找到所有距离最小的点
    nearest_indices = distances[distances == min_distance].index
    nearest_data = data.loc[nearest_indices]
    
    if len(nearest_data) == 1:
        matched_point = nearest_data.iloc[0]
        # 直接显示最近点的坐标，而不是距离信息
        nearest_info = f"({matched_point['recall']:.3f},{matched_point['coverage']:.3f})"
        return matched_point['error'], nearest_info, matched_point['recall'], matched_point['coverage']
    else:
        # 如果有多个距离相等的点，优先选择recall更接近的点
        recall_distances = abs(nearest_data['recall'] - recall)
        min_recall_distance = recall_distances.min()
        
        # 选择recall距离最小的点
        best_matches = nearest_data[recall_distances == min_recall_distance]
        
        # 如果还有多个点，选择第一个并返回其坐标
        best_match = best_matches.iloc[0]
        # 直接显示最近点的坐标，而不是距离信息
        nearest_info = f"({best_match['recall']:.3f},{best_match['coverage']:.3f})"
        return best_matches['error'].mean(), nearest_info, best_match['recall'], best_match['coverage']


def predict_single_point(model, recall, coverage, show_features=False):
    """
    预测单个(recall, coverage)点的error值
    
    Args:
        model: 训练好的LightGBM模型
        recall: recall值
        coverage: coverage值
        show_features: 是否显示生成的特征
    
    Returns:
        预测的error值
    """
    # 创建输入数据
    input_data = pd.DataFrame([[recall, coverage]], columns=['recall', 'coverage'])
    
    # 生成增强特征
    enhanced_data = enhance_features_for_prediction(input_data)
    feature_columns = get_feature_columns_for_prediction()
    
    # 确保特征顺序正确
    X_enhanced = enhanced_data[feature_columns]
    
    if show_features:
        print(f"\n生成的特征 (recall={recall:.3f}, coverage={coverage:.3f}):")
        print("-" * 50)
        for i, (col, val) in enumerate(zip(feature_columns, X_enhanced.iloc[0])):
            print(f"{i+1:2d}. {col:<25}: {val:.6f}")
        print("-" * 50)
    
    # 预测
    prediction = model.predict(X_enhanced)[0]
    
    return prediction


def predict_batch_points(model, recall_coverage_pairs, show_summary=True):
    """
    批量预测多个(recall, coverage)点的error值
    
    Args:
        model: 训练好的LightGBM模型
        recall_coverage_pairs: (recall, coverage)对的列表
        show_summary: 是否显示预测摘要
    
    Returns:
        预测值列表
    """
    print(f"开始批量预测 {len(recall_coverage_pairs)} 个点...")
    
    # 创建输入数据
    input_data = pd.DataFrame(recall_coverage_pairs, columns=['recall', 'coverage'])
    
    # 生成增强特征
    enhanced_data = enhance_features_for_prediction(input_data)
    feature_columns = get_feature_columns_for_prediction()
    
    # 确保特征顺序正确
    X_enhanced = enhanced_data[feature_columns]
    
    print(f"✓ 特征验证: 模型期望 {model.num_feature()} 个特征，提供 {X_enhanced.shape[1]} 个特征")
    
    if model.num_feature() != X_enhanced.shape[1]:
        raise ValueError(f"特征数量不匹配！模型需要 {model.num_feature()} 个特征，但提供了 {X_enhanced.shape[1]} 个特征")
    
    # 批量预测
    predictions = model.predict(X_enhanced)
    
    if show_summary:
        print(f"✓ 预测完成")
        print(f"  预测值范围: [{predictions.min():.6f}, {predictions.max():.6f}]")
        print(f"  预测值平均: {predictions.mean():.6f}")
    
    return predictions


def apply_monotonic_correction(model, recall_coverage_pairs):
    """
    对批量预测结果应用单调性校正
    
    Args:
        model: 训练好的LightGBM模型
        recall_coverage_pairs: (recall, coverage)对的列表
    
    Returns:
        校正后的预测值列表
    """
    # 先进行原始预测
    raw_predictions = predict_batch_points(model, recall_coverage_pairs, show_summary=False)
    
    # 组织数据为矩阵形式进行校正
    recall_values = [pair[0] for pair in recall_coverage_pairs]
    coverage_values = [pair[1] for pair in recall_coverage_pairs]
    
    unique_recalls = sorted(list(set(recall_values)))
    unique_coverages = sorted(list(set(coverage_values)))
    
    print(f"\n应用单调性校正...")
    print(f"矩阵维度: {len(unique_recalls)} recalls × {len(unique_coverages)} coverages")
    
    # 创建预测值矩阵
    error_matrix = np.full((len(unique_recalls), len(unique_coverages)), np.nan)
    index_matrix = np.full((len(unique_recalls), len(unique_coverages)), -1, dtype=int)
    
    # 填充矩阵
    for i, (recall, coverage) in enumerate(recall_coverage_pairs):
        r_idx = unique_recalls.index(recall)
        c_idx = unique_coverages.index(coverage)
        error_matrix[r_idx, c_idx] = raw_predictions[i]
        index_matrix[r_idx, c_idx] = i
    
    # 校正矩阵 - 多次迭代直到收敛
    corrected_matrix = error_matrix.copy()
    max_iterations = 10
    
    for iteration in range(max_iterations):
        changed = False
        
        # 行内单调性校正（相同recall，coverage增加 → error增加）
        for r_idx in range(len(unique_recalls)):
            row = corrected_matrix[r_idx, :]
            valid_indices = ~np.isnan(row)
            
            if np.sum(valid_indices) > 1:
                valid_cols = np.where(valid_indices)[0]
                for i in range(len(valid_cols) - 1):
                    curr_col = valid_cols[i]
                    next_col = valid_cols[i + 1]
                    
                    if row[next_col] < row[curr_col]:
                        corrected_matrix[r_idx, next_col] = row[curr_col]
                        changed = True
        
        # 列内单调性校正（相同coverage，recall增加 → error增加）
        for c_idx in range(len(unique_coverages)):
            col = corrected_matrix[:, c_idx]
            valid_indices = ~np.isnan(col)
            
            if np.sum(valid_indices) > 1:
                valid_rows = np.where(valid_indices)[0]
                for i in range(len(valid_rows) - 1):
                    curr_row = valid_rows[i]
                    next_row = valid_rows[i + 1]
                    
                    if col[next_row] < col[curr_row]:
                        corrected_matrix[next_row, c_idx] = col[curr_row]
                        changed = True
        
        if not changed:
            print(f"✓ 单调性校正收敛于第{iteration+1}次迭代")
            break
    
    # 将校正后的值映射回原始预测数组
    corrected_predictions = raw_predictions.copy()
    for r_idx in range(len(unique_recalls)):
        for c_idx in range(len(unique_coverages)):
            if index_matrix[r_idx, c_idx] != -1:
                orig_idx = index_matrix[r_idx, c_idx]
                corrected_predictions[orig_idx] = corrected_matrix[r_idx, c_idx]
    
    return corrected_predictions


def main():
    """主函数 - 演示增强特征模型预测"""
    
    # 配置路径
    model_path = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_enhanced/model_quantile_0.95_enhanced.txt"
    original_data_path = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented/filter_58_triples_augmented.csv"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练增强特征模型")
        return 1
    
    # 加载模型
    try:
        model = lgb.Booster(model_file=model_path)
        print(f"✅ 增强特征模型已加载: {model_path}")
        print(f"   模型使用 {model.num_feature()} 个特征")
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return 1
    
    # 加载原始数据用于对比
    original_data = load_original_data(original_data_path)
    
    print("\n" + "="*80)
    print("增强特征模型预测演示")
    print("="*80)
    
    # 演示1: 单点预测
    print("\n1. 单点预测演示")
    print("-" * 40)
    
    test_recall = 0.95
    test_coverage = 0.9
    
    prediction = predict_single_point(model, test_recall, test_coverage, show_features=True)
    
    # 查找实际值
    actual_error, match_type, actual_recall, actual_coverage = find_actual_error_with_info(
        original_data, test_recall, test_coverage)
    
    print(f"\n单点预测结果:")
    print(f"输入: recall={test_recall:.3f}, coverage={test_coverage:.3f}")
    print(f"预测error: {prediction:.6f}")
    if not np.isnan(actual_error):
        print(f"实际error: {actual_error:.6f} ({match_type})")
        print(f"预测差异: {prediction - actual_error:+.6f}")
    else:
        print(f"实际error: 无数据")
    
    # 演示2: 批量预测
    print("\n\n2. 批量预测演示")
    print("-" * 40)
    
    # 定义测试点
    recall_values = [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
    coverage_values = [0.85, 0.9, 0.95, 0.99, 1.0]
    
    # 生成所有组合
    test_points = []
    for recall in recall_values:
        for coverage in coverage_values:
            test_points.append([recall, coverage])
    
    print(f"预测 {len(test_points)} 个 (recall, coverage) 组合...")
    
    # 批量预测（带单调性校正）
    corrected_predictions = apply_monotonic_correction(model, test_points)
    
    # 显示结果表格
    print(f"\n预测结果表格 (增强特征模型):")
    print("-" * 100)
    print(f"{'Recall':<8} | {'Coverage':<8} | {'Predicted':<12} | {'Actual':<12} | {'Diff':<12} | {'Match Type':<15}")
    print("-" * 100)
    
    for i, (recall, coverage) in enumerate(test_points):
        prediction = corrected_predictions[i]
        
        # 查找实际值
        actual_error, match_type, actual_recall, actual_coverage = find_actual_error_with_info(
            original_data, recall, coverage)
        
        if np.isnan(actual_error):
            diff_str = "N/A"
            actual_str = "NaN"
        else:
            diff = prediction - actual_error
            diff_str = f"{diff:+.6f}"
            actual_str = f"{actual_error:.6f}"
        
        print(f"{recall:<8.3f} | {coverage:<8.3f} | {prediction:<12.6f} | {actual_str:<12} | {diff_str:<12} | {match_type:<15}")
    
    # 演示3: 矩阵表格显示
    print(f"\n\n3. 矩阵表格显示")
    print("-" * 40)
    
    print(f"预测结果矩阵 (行=recall, 列=coverage):")
    print("Recall\\Coverage", end="")
    for coverage in coverage_values:
        print(f"{coverage:>10.3f}", end="")
    print()
    print("-" * (15 + 10 * len(coverage_values)))
    
    idx = 0
    for recall in recall_values:
        print(f"{recall:>8.3f}      ", end="")
        for coverage in coverage_values:
            prediction = corrected_predictions[idx]
            print(f"{prediction:>10.6f}", end="")
            idx += 1
        print()
    
    # 统计信息
    if original_data is not None:
        print(f"\n\n4. 预测准确性统计")
        print("-" * 40)
        
        valid_predictions = []
        valid_actuals = []
        
        for i, (recall, coverage) in enumerate(test_points):
            prediction = corrected_predictions[i]
            actual_error, _, _, _ = find_actual_error_with_info(original_data, recall, coverage)
            
            if not np.isnan(actual_error):
                valid_predictions.append(prediction)
                valid_actuals.append(actual_error)
        
        if valid_predictions:
            valid_predictions = np.array(valid_predictions)
            valid_actuals = np.array(valid_actuals)
            
            mae = np.mean(np.abs(valid_predictions - valid_actuals))
            rmse = np.sqrt(np.mean((valid_predictions - valid_actuals) ** 2))
            safe_ratio = np.mean(valid_predictions >= valid_actuals)
            
            print(f"有效对比点数量: {len(valid_predictions)}/{len(test_points)}")
            print(f"平均绝对误差 (MAE): {mae:.6f}")
            print(f"均方根误差 (RMSE): {rmse:.6f}")
            print(f"安全预测比例 (预测≥实际): {safe_ratio:.2%}")
            print(f"预测值范围: [{valid_predictions.min():.6f}, {valid_predictions.max():.6f}]")
            print(f"实际值范围: [{valid_actuals.min():.6f}, {valid_actuals.max():.6f}]")
    
    print(f"\n" + "="*80)
    print("预测完成！")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 