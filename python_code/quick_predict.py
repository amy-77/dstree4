#!/usr/bin/env python3
"""
快速预测指定recall值的error，并与原始数据对比
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
import os
import sys


def load_original_data(data_path):
    """加载原始训练数据"""
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
        return matched_point['error'], f"Nearest(d={min_distance:.4f})", matched_point['recall'], matched_point['coverage']
    else:
        # 如果有多个距离相等的点，优先选择recall更接近的点
        recall_distances = abs(nearest_data['recall'] - recall)
        min_recall_distance = recall_distances.min()
        
        # 选择recall距离最小的点
        best_matches = nearest_data[recall_distances == min_recall_distance]
        
        # 如果还有多个点，选择第一个并返回其坐标
        best_match = best_matches.iloc[0]
        return best_matches['error'].mean(), f"Nearest(d={min_distance:.4f})", best_match['recall'], best_match['coverage']


def find_actual_error(data, recall, coverage, tolerance=0.001):
    """在原始数据中查找对应的实际error值"""
    result, _, _, _ = find_actual_error_with_info(data, recall, coverage, tolerance)
    return result


def apply_monotonic_correction_sorted(model, X):
    """
    基于矩阵的单调性校正算法 - 将(recall, coverage)对组织成矩阵进行校正
    """
    raw_predictions = model.predict(X)
    
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X
        
    if X_array.ndim == 1:
        X_array = X_array.reshape(1, -1)
        raw_predictions = np.array([raw_predictions])
    
    n = len(X_array)
    
    # 1. 提取所有唯一的recall和coverage值并排序
    unique_recalls = sorted(np.unique(X_array[:, 0]))
    unique_coverages = sorted(np.unique(X_array[:, 1]))
    
    print(f"矩阵维度: {len(unique_recalls)} recalls × {len(unique_coverages)} coverages")
    print(f"Recalls: {[f'{r:.3f}' for r in unique_recalls]}")
    print(f"Coverages: {[f'{c:.3f}' for c in unique_coverages]}")
    
    # 2. 创建矩阵存储预测值，初始化为NaN
    error_matrix = np.full((len(unique_recalls), len(unique_coverages)), np.nan)
    index_matrix = np.full((len(unique_recalls), len(unique_coverages)), -1, dtype=int)
    
    # 3. 填充矩阵
    for i, (recall, coverage) in enumerate(X_array):
        r_idx = unique_recalls.index(recall)
        c_idx = unique_coverages.index(coverage)
        error_matrix[r_idx, c_idx] = raw_predictions[i]
        index_matrix[r_idx, c_idx] = i
    
    print(f"原始预测矩阵:")
    print("Recall\\Coverage", end="")
    for c in unique_coverages:
        print(f"{c:>8.3f}", end="")
    print()
    for r_idx, r in enumerate(unique_recalls):
        print(f"{r:>8.3f}       ", end="")
        for c_idx in range(len(unique_coverages)):
            if not np.isnan(error_matrix[r_idx, c_idx]):
                print(f"{error_matrix[r_idx, c_idx]:>8.3f}", end="")
            else:
                print(f"{'---':>8}", end="")
        print()
    
    # 4. 校正矩阵 - 多次迭代直到收敛
    corrected_matrix = error_matrix.copy()
    max_iterations = 10
    
    for iteration in range(max_iterations):
        changed = False
        old_matrix = corrected_matrix.copy()
        
        # 4.1 行内单调性校正（相同recall，coverage增加 → error增加）
        for r_idx in range(len(unique_recalls)):
            row = corrected_matrix[r_idx, :]
            valid_indices = ~np.isnan(row)
            
            if np.sum(valid_indices) > 1:
                valid_cols = np.where(valid_indices)[0]
                for i in range(len(valid_cols) - 1):
                    curr_col = valid_cols[i]
                    next_col = valid_cols[i + 1]
                    
                    if row[next_col] < row[curr_col]:
                        old_val = row[next_col]
                        corrected_matrix[r_idx, next_col] = row[curr_col]
                        changed = True
                        print(f"行内校正: recall={unique_recalls[r_idx]:.3f}, "
                              f"coverage {unique_coverages[curr_col]:.3f}→{unique_coverages[next_col]:.3f}, "
                              f"error {old_val:.6f}→{row[curr_col]:.6f}")
        
        # 4.2 列内单调性校正（相同coverage，recall增加 → error增加）
        for c_idx in range(len(unique_coverages)):
            col = corrected_matrix[:, c_idx]
            valid_indices = ~np.isnan(col)
            
            if np.sum(valid_indices) > 1:
                valid_rows = np.where(valid_indices)[0]
                for i in range(len(valid_rows) - 1):
                    curr_row = valid_rows[i]
                    next_row = valid_rows[i + 1]
                    
                    if col[next_row] < col[curr_row]:
                        old_val = col[next_row]
                        corrected_matrix[next_row, c_idx] = col[curr_row]
                        changed = True
                        print(f"列内校正: coverage={unique_coverages[c_idx]:.3f}, "
                              f"recall {unique_recalls[curr_row]:.3f}→{unique_recalls[next_row]:.3f}, "
                              f"error {old_val:.6f}→{col[curr_row]:.6f}")
        
        if not changed:
            print(f"单调性校正收敛于第{iteration+1}次迭代")
            break
        else:
            print(f"第{iteration+1}次迭代完成")
    
    print(f"校正后预测矩阵:")
    print("Recall\\Coverage", end="")
    for c in unique_coverages:
        print(f"{c:>8.3f}", end="")
    print()
    for r_idx, r in enumerate(unique_recalls):
        print(f"{r:>8.3f}       ", end="")
        for c_idx in range(len(unique_coverages)):
            if not np.isnan(corrected_matrix[r_idx, c_idx]):
                print(f"{corrected_matrix[r_idx, c_idx]:>8.3f}", end="")
            else:
                print(f"{'---':>8}", end="")
        print()
    
    # 5. 将校正后的值映射回原始预测数组
    corrected_predictions = raw_predictions.copy()
    for r_idx in range(len(unique_recalls)):
        for c_idx in range(len(unique_coverages)):
            if index_matrix[r_idx, c_idx] != -1:
                orig_idx = index_matrix[r_idx, c_idx]
                corrected_predictions[orig_idx] = corrected_matrix[r_idx, c_idx]
    
    return corrected_predictions[0] if len(corrected_predictions) == 1 else corrected_predictions


def quick_predict():
    """快速预测指定recall值的error，并与原始数据对比"""
    
    # 用户指定的recall值
    recall_values = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.999]
    # 常用的coverage值
    coverage_values = [0.85, 0.9, 0.95, 0.99]
    
    # 模型和数据路径
    # model_dir = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models/"
    # model_path = os.path.join(model_dir, "model_quantile_0.9.txt") #  model_quantile_0.9_mono_1_1.txt

    # 增强模型
    model_dir = "/home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models/"
    model_path = os.path.join(model_dir, "model_quantile_0.9.txt") 
    # dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models/model_quantile_0.9.txt
    #home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_enhanced/model_quantile_0.9_enhanced.txt
    data_path = "/home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/lgbm_data/filter_549_triples.csv"
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型或指定正确的模型路径")
        return 1
    
    # 加载模型
    try:
        model = lgb.Booster(model_file=model_path)
        print(f"✓ 模型已加载: {model_path}")
    except Exception as e:
        print(f"✗ 加载模型失败: {e}")
        return 1
    
    # 加载原始数据
    original_data = load_original_data(data_path)
    
    # 准备所有(recall, coverage)组合进行批量预测和校正
    print("\n" + "="*60)
    print("准备批量预测所有(recall, coverage)组合...")
    print("="*60)
    
    all_combinations = []
    for recall in recall_values:
        for coverage in coverage_values:
            all_combinations.append([recall, coverage])
    
    X_all = np.array(all_combinations)
    print(f"总共 {len(all_combinations)} 个(recall, coverage)组合")
    
    # 一次性进行矩阵单调性校正
    print("\n" + "="*60)
    print("正在进行矩阵单调性校正...")
    print("="*60)
    corrected_predictions = apply_monotonic_correction_sorted(model, X_all)
    
    print("\n" + "="*90)
    print("预测结果：不同recall和coverage组合的error值 (预测 vs 实际)")
    print("="*90)
    
    # 表头
    print(f"{'Recall':<8} |", end="")
    for coverage in coverage_values:
        print(f" Cov={coverage:<5.3f}(P/A) |", end="")
    print()
    print("-" * 90)
    
    # 为每个recall值显示所有coverage的预测结果
    idx = 0
    for recall in recall_values:
        print(f"{recall:<8.3f} |", end="")
        for coverage in coverage_values:
            # 获取校正后的预测值
            prediction = corrected_predictions[idx]
            
            # 查找实际error
            actual_error, match_type, actual_recall, actual_coverage = find_actual_error_with_info(original_data, recall, coverage)
            
            if np.isnan(actual_error):
                print(f" {prediction:<6.3f}/{match_type}   |", end="")
            else:
                print(f" {prediction:<6.3f}/{actual_error:<6.3f} |", end="")
            
            idx += 1
        print()
    
    print("="*90)
    print("说明: P=预测值, A=实际值, NoData=原始数据中无对应点")
    print("      匹配类型: Exact=精确匹配, Nearest=最近邻匹配")
    
    # 详细对比表格
    print("\n所有recall和coverage组合详细对比:")
    print("-" * 140)
    print(f"{'Recall':<8} | {'Coverage':<8} | {'Predicted':<12} | {'Actual':<12} | {'Diff':<12} | {'Match Type':<15} | {'Act_R':<8} | {'Act_C':<8}")
    print("-" * 140)
    
    idx = 0
    for recall in recall_values:
        for coverage in coverage_values:
            # 获取校正后的预测值
            prediction = corrected_predictions[idx]
            
            # 查找实际error
            actual_error, match_type, actual_recall, actual_coverage = find_actual_error_with_info(original_data, recall, coverage)
            
            if np.isnan(actual_error):
                diff_str = "N/A"
                actual_str = "NaN"
                actual_r_str = "N/A"
                actual_c_str = "N/A"
            else:
                diff = prediction - actual_error
                diff_str = f"{diff:+.6f}"
                actual_str = f"{actual_error:.6f}"
                actual_r_str = f"{actual_recall:.3f}"
                actual_c_str = f"{actual_coverage:.3f}"
            
            print(f"{recall:<8.3f} | {coverage:<8.3f} | {prediction:<12.6f} | {actual_str:<12} | {diff_str:<12} | {match_type:<15} | {actual_r_str:<8} | {actual_c_str:<8}")
            
            idx += 1
    
    # 统计信息
    if original_data is not None:
        print("\n数据统计:")
        print("-" * 40)
        
        # 计算所有有实际数据的点的预测误差
        valid_predictions = []
        valid_actuals = []
        
        idx = 0
        for recall in recall_values:
            for coverage in coverage_values:
                prediction = corrected_predictions[idx]
                actual_error, _, _, _ = find_actual_error_with_info(original_data, recall, coverage)
                
                if not np.isnan(actual_error):
                    valid_predictions.append(prediction)
                    valid_actuals.append(actual_error)
                
                idx += 1
        
        if valid_predictions:
            valid_predictions = np.array(valid_predictions)
            valid_actuals = np.array(valid_actuals)
            
            mae = np.mean(np.abs(valid_predictions - valid_actuals))
            rmse = np.sqrt(np.mean((valid_predictions - valid_actuals) ** 2))
            safe_ratio = np.mean(valid_predictions >= valid_actuals)
            
            print(f"有效对比点数量: {len(valid_predictions)}")
            print(f"平均绝对误差 (MAE): {mae:.6f}")
            print(f"均方根误差 (RMSE): {rmse:.6f}")
            print(f"安全预测比例: {safe_ratio:.2%}")
            print(f"预测值范围: [{valid_predictions.min():.6f}, {valid_predictions.max():.6f}]")
            print(f"实际值范围: [{valid_actuals.min():.6f}, {valid_actuals.max():.6f}]")
    
    return 0


if __name__ == "__main__":
    sys.exit(quick_predict()) 