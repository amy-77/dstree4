#!/usr/bin/env python3
"""
对所有单独训练的模型进行批量预测，在关键点进行预测并包含单调性校正
同时统计原始数据中的actual_error值
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import sys


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
    
    # 2. 创建矩阵存储预测值，初始化为NaN
    error_matrix = np.full((len(unique_recalls), len(unique_coverages)), np.nan)
    index_matrix = np.full((len(unique_recalls), len(unique_coverages)), -1, dtype=int)
    
    # 3. 填充矩阵
    for i, (recall, coverage) in enumerate(X_array):
        r_idx = unique_recalls.index(recall)
        c_idx = unique_coverages.index(coverage)
        error_matrix[r_idx, c_idx] = raw_predictions[i]
        index_matrix[r_idx, c_idx] = i
    
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
                        corrected_matrix[r_idx, next_col] = row[curr_col]
                        changed = True
        
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
                        corrected_matrix[next_row, c_idx] = col[curr_row]
                        changed = True
        
        if not changed:
            break
    
    # 5. 将校正后的值映射回原始预测数组
    corrected_predictions = raw_predictions.copy()
    for r_idx in range(len(unique_recalls)):
        for c_idx in range(len(unique_coverages)):
            if index_matrix[r_idx, c_idx] != -1:
                orig_idx = index_matrix[r_idx, c_idx]
                corrected_predictions[orig_idx] = corrected_matrix[r_idx, c_idx]
    
    return corrected_predictions[0] if len(corrected_predictions) == 1 else corrected_predictions


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


def load_original_data(data_path):
    """加载原始训练数据"""
    if not os.path.exists(data_path):
        print(f"原始数据文件不存在: {data_path}")
        return None
    
    try:
        data = pd.read_csv(data_path)
        return data
    except Exception as e:
        print(f"✗ 加载原始数据失败: {e}")
        return None


def predict_with_model_monotonic(recall_values, coverage_values, model_path):
    """使用单个LightGBM模型进行预测，包含单调性校正
    Args:
        recall_values: recall值列表
        coverage_values: coverage值列表  
        model_path: 模型文件路径
    Returns:
        预测结果字典，键为(recall, coverage)，值为预测error
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型
    model = lgb.Booster(model_file=model_path)
    
    # 准备所有(recall, coverage)组合
    all_combinations = []
    for recall in recall_values:
        for coverage in coverage_values:
            all_combinations.append([recall, coverage])
    
    X_all = np.array(all_combinations)
    
    # 进行单调性校正预测
    corrected_predictions = apply_monotonic_correction_sorted(model, X_all)
    
    # 构建结果字典
    results = {}
    idx = 0
    for recall in recall_values:
        for coverage in coverage_values:
            results[(recall, coverage)] = corrected_predictions[idx]
            idx += 1
            
    return results


def get_all_filter_ids(base_dir, model_filename="model_quantile_0.9.txt"):
    """获取所有filter ID"""
    filter_ids = []
    for item in os.listdir(base_dir):
        if item.startswith("filter_") and os.path.isdir(os.path.join(base_dir, item)):
            try:
                filter_id = int(item.split("_")[1])
                # 检查是否存在模型文件
                model_path = os.path.join(base_dir, item, model_filename)
                if os.path.exists(model_path):
                    filter_ids.append(filter_id)
            except ValueError:
                continue
    return sorted(filter_ids)


def batch_predict_key_points():
    """对所有模型进行批量预测"""
    
    # 配置参数  quantile_models_individual_95
    base_dir = "/home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models_individual_original"
    output_dir = "/home/qwang/projects/leafi/dstree2/deep1b_result_model_original/key_points_predictions_without_quantile_original"
    model_filename = "model_mono_1_1.txt"
    
    # 原始数据目录（假设数据结构与训练时一致）
    data_base_dir = "/home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/lgbm_data"
    
    # 要测试的recall和coverage组合（与quick_predict.py保持一致）
    recall_values = [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 0.999, 1]
    coverage_values = [0.85, 0.9, 0.95, 0.99, 1]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有filter ID
    filter_ids = get_all_filter_ids(base_dir, model_filename)
    print(f"找到 {len(filter_ids)} 个filter模型")
    print(f"模型目录: {base_dir}")
    print(f"数据目录: {data_base_dir}")
    print(f"输出目录: {output_dir}")
    print(f"模型文件名: {model_filename}")
    
    # 存储所有预测结果和实际结果，按recall-coverage组合分组
    predictions_by_combination = defaultdict(list)
    actuals_by_combination = defaultdict(list)
    
    # 对每个filter进行推理
    successful_filters = []
    failed_filters = []
    
    for i, filter_id in enumerate(filter_ids):
        model_path = os.path.join(base_dir, f"filter_{filter_id}", model_filename)
        data_path = os.path.join(data_base_dir, f"filter_{filter_id}_triples.csv")
        
        print(f"处理Filter {filter_id} ({i+1}/{len(filter_ids)})...")
        
        try:
            # 批量预测所有recall-coverage组合（包含单调性校正）
            predictions = predict_with_model_monotonic(recall_values, coverage_values, model_path)
            
            # 加载原始数据
            original_data = load_original_data(data_path)
            
            # 将结果添加到对应组合中
            for (recall, coverage), predicted_error in predictions.items():
                # 添加预测结果
                predictions_by_combination[(recall, coverage)].append({
                    'Filter_ID': filter_id,
                    'Recall': recall,
                    'Coverage': coverage,
                    'Predicted_Error': predicted_error
                })
                
                # 查找并添加实际error
                actual_error = find_actual_error(original_data, recall, coverage)
                actuals_by_combination[(recall, coverage)].append({
                    'Filter_ID': filter_id,
                    'Recall': recall,
                    'Coverage': coverage,
                    'Actual_Error': actual_error
                })
            
            successful_filters.append(filter_id)
            print(f"  ✓ Filter {filter_id} 处理完成")
            
        except Exception as e:
            failed_filters.append(filter_id)
            print(f"  ✗ Filter {filter_id} 处理失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n成功处理 {len(successful_filters)} 个filter")
    print(f"失败 {len(failed_filters)} 个filter")
    if failed_filters:
        print(f"失败的filter: {failed_filters}")
    
    # 保存每个recall-coverage组合的预测结果和实际结果
    print("\n保存预测结果文件...")
    for (recall, coverage), predictions in predictions_by_combination.items():
        if predictions:
            # 合并预测结果和实际结果
            actuals = actuals_by_combination[(recall, coverage)]
            
            # 创建完整的数据框
            combined_data = []
            for pred, actual in zip(predictions, actuals):
                combined_data.append({
                    'Filter_ID': pred['Filter_ID'],
                    'Recall': pred['Recall'],
                    'Coverage': pred['Coverage'],
                    'Predicted_Error': pred['Predicted_Error'],
                    'Actual_Error': actual['Actual_Error']
                })
            
            df = pd.DataFrame(combined_data)
            df = df.sort_values('Filter_ID')
            
            filename = f"pred_errors_recall{recall:.3f}_cov{coverage:.3f}.txt"
            output_path = os.path.join(output_dir, filename)
            df.to_csv(output_path, index=False)
            print(f"  ✓ 保存 {filename}: {len(df)} 个预测结果")
    
    # 计算每个recall-coverage组合的统计信息（包括预测值和实际值的对比）
    print("\n计算统计信息...")
    stats_results = []
    
    for (recall, coverage), predictions in predictions_by_combination.items():
        if predictions:
            pred_df = pd.DataFrame(predictions)
            actual_df = pd.DataFrame(actuals_by_combination[(recall, coverage)])
            
            predicted_errors = pred_df['Predicted_Error'].values
            actual_errors = actual_df['Actual_Error'].values
            
            # 过滤掉NaN的实际值
            valid_mask = ~np.isnan(actual_errors)
            valid_predicted = predicted_errors[valid_mask]
            valid_actual = actual_errors[valid_mask]
            
            stats_result = {
                'Recall': recall,
                'Coverage': coverage,
                'Count': len(predicted_errors),
                'Valid_Actual_Count': len(valid_actual),
                # 预测值统计
                'Mean_Predicted': np.mean(predicted_errors),
                'Std_Predicted': np.std(predicted_errors),
                'Variance_Predicted': np.var(predicted_errors),
                'Min_Predicted': np.min(predicted_errors),
                'Max_Predicted': np.max(predicted_errors),
                'Median_Predicted': np.median(predicted_errors),
                # 实际值统计
                'Mean_Actual': np.mean(valid_actual) if len(valid_actual) > 0 else np.nan,
                'Std_Actual': np.std(valid_actual) if len(valid_actual) > 0 else np.nan,
                'Variance_Actual': np.var(valid_actual) if len(valid_actual) > 0 else np.nan,
                'Min_Actual': np.min(valid_actual) if len(valid_actual) > 0 else np.nan,
                'Max_Actual': np.max(valid_actual) if len(valid_actual) > 0 else np.nan,
                'Median_Actual': np.median(valid_actual) if len(valid_actual) > 0 else np.nan
            }
            
            # 预测vs实际对比统计
            if len(valid_actual) > 0:
                mae = np.mean(np.abs(valid_predicted - valid_actual))
                rmse = np.sqrt(np.mean((valid_predicted - valid_actual) ** 2))
                safe_ratio = np.mean(valid_predicted >= valid_actual)
                
                stats_result.update({
                    'MAE': mae,
                    'RMSE': rmse,
                    'Safe_Prediction_Ratio': safe_ratio
                })
            else:
                stats_result.update({
                    'MAE': np.nan,
                    'RMSE': np.nan,
                    'Safe_Prediction_Ratio': np.nan
                })
            
            stats_results.append(stats_result)
    
    # 保存统计信息
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_df = stats_df.sort_values(['Recall', 'Coverage'])
        
        stats_output_path = os.path.join(output_dir, "prediction_statistics.csv")
        stats_df.to_csv(stats_output_path, index=False)
        print(f"  ✓ 保存统计信息: {stats_output_path}")
        
        # 显示统计结果
        print("\n预测vs实际统计信息:")
        print("=" * 150)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(stats_df.to_string(index=False, float_format='%.6f'))
        
        # 创建关键指标的交叉表格式
        # 平均预测误差交叉表
        mean_pred_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='Mean_Predicted')
        print("\n平均预测误差交叉表:")
        print("=" * 60)
        print(mean_pred_pivot.to_string(float_format='{:.6f}'.format))
        
        # 平均实际误差交叉表
        mean_actual_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='Mean_Actual')
        print("\n平均实际误差交叉表:")
        print("=" * 60)
        print(mean_actual_pivot.to_string(float_format='{:.6f}'.format))
        
        # 实际误差方差交叉表
        var_actual_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='Variance_Actual')
        print("\n实际误差方差交叉表:")
        print("=" * 60)
        print(var_actual_pivot.to_string(float_format='{:.6f}'.format))
        
        # MAE交叉表
        mae_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='MAE')
        print("\nMAE交叉表:")
        print("=" * 60)
        print(mae_pivot.to_string(float_format='{:.6f}'.format))
        
        # 安全预测比例交叉表
        safe_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='Safe_Prediction_Ratio')
        print("\n安全预测比例交叉表:")
        print("=" * 60)
        print(safe_pivot.to_string(float_format='{:.3f}'.format))
        
        # 预测误差方差交叉表
        var_pred_pivot = stats_df.pivot(index='Recall', columns='Coverage', values='Variance_Predicted')
        print("\n预测误差方差交叉表:")
        print("=" * 60)
        print(var_pred_pivot.to_string(float_format='{:.6f}'.format))
        
        # 保存交叉表
        mean_pred_pivot.to_csv(os.path.join(output_dir, "mean_predicted_errors_table.csv"))
        mean_actual_pivot.to_csv(os.path.join(output_dir, "mean_actual_errors_table.csv"))
        var_actual_pivot.to_csv(os.path.join(output_dir, "variance_actual_errors_table.csv"))
        mae_pivot.to_csv(os.path.join(output_dir, "mae_table.csv"))
        safe_pivot.to_csv(os.path.join(output_dir, "safe_prediction_ratio_table.csv"))
        var_pred_pivot.to_csv(os.path.join(output_dir, "variance_predicted_errors_table.csv"))
        
        print(f"\n✓ 保存各项交叉表到输出目录")
        print(f"✓ 实际误差方差交叉表: variance_actual_errors_table.csv")
        print(f"✓ 预测误差方差交叉表: variance_predicted_errors_table.csv")
    
    print(f"\n批量预测完成！结果保存在: {output_dir}")
    print(f"总共处理了 {len(recall_values) * len(coverage_values)} 个recall-coverage组合")
    return len(successful_filters), len(failed_filters)


if __name__ == "__main__":
    try:
        successful, failed = batch_predict_key_points()
        print(f"\n总结: 成功 {successful} 个filter, 失败 {failed} 个filter")
    except Exception as e:
        print(f"批量预测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 