#!/usr/bin/env python3
import lightgbm as lgb
import pandas as pd
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp


def enhance_features(data):
    """
    对数据进行特征工程，增加新的特征来提高模型表达能力
    
    Args:
        data: 包含 recall 和 coverage 列的 DataFrame
    
    Returns:
        增强后的 DataFrame
    """
    print("正在进行特征工程...")
    
    # 避免修改原始数据
    enhanced_data = data.copy()
    
    # 1. 交互特征
    enhanced_data['recall_coverage_product'] = enhanced_data['recall'] * enhanced_data['coverage']
    enhanced_data['recall_coverage_ratio'] = enhanced_data['recall'] / (enhanced_data['coverage'] + 1e-10)
    enhanced_data['coverage_recall_ratio'] = enhanced_data['coverage'] / (enhanced_data['recall'] + 1e-10)
    
    # 2. 多项式特征
    enhanced_data['recall_squared'] = enhanced_data['recall'] ** 2
    enhanced_data['coverage_squared'] = enhanced_data['coverage'] ** 2
    enhanced_data['recall_cubed'] = enhanced_data['recall'] ** 3
    enhanced_data['coverage_cubed'] = enhanced_data['coverage'] ** 3
    
    # 3. 对数特征（提高极值区域敏感性）
    enhanced_data['log_recall'] = np.log(enhanced_data['recall'] + 1e-10)
    enhanced_data['log_coverage'] = np.log(enhanced_data['coverage'] + 1e-10)
    enhanced_data['log_product'] = enhanced_data['log_recall'] + enhanced_data['log_coverage']
    
    # 4. 距离特征
    enhanced_data['distance_to_perfect'] = np.sqrt((1-enhanced_data['recall'])**2 + (1-enhanced_data['coverage'])**2)
    enhanced_data['distance_to_origin'] = np.sqrt(enhanced_data['recall']**2 + enhanced_data['coverage']**2)
    enhanced_data['manhattan_distance_to_perfect'] = (1-enhanced_data['recall']) + (1-enhanced_data['coverage'])
    
    # 5. 关系特征（替代三角函数）
    # 计算recall和coverage的"协调性" - 当两者都高时值较大，当差异很大时值较小
    enhanced_data['harmony_score'] = 1 - np.abs(enhanced_data['recall'] - enhanced_data['coverage'])
    enhanced_data['balance_score'] = np.minimum(enhanced_data['recall'], enhanced_data['coverage']) / np.maximum(enhanced_data['recall'], enhanced_data['coverage'])
    
    # 局部相关性指标 - 基于排序位置的相关性
    if len(enhanced_data) > 1:
        # 计算每个点在其邻域内的recall-coverage关系
        enhanced_data['recall_rank'] = enhanced_data['recall'].rank(pct=True)
        enhanced_data['coverage_rank'] = enhanced_data['coverage'].rank(pct=True)
        enhanced_data['rank_correlation'] = enhanced_data['recall_rank'] * enhanced_data['coverage_rank']
        enhanced_data['rank_difference'] = np.abs(enhanced_data['recall_rank'] - enhanced_data['coverage_rank'])
    else:
        enhanced_data['recall_rank'] = 0.5
        enhanced_data['coverage_rank'] = 0.5
        enhanced_data['rank_correlation'] = 0.25
        enhanced_data['rank_difference'] = 0.0
    
    # 6. 指数特征
    enhanced_data['exp_recall'] = np.exp(-enhanced_data['recall'])
    enhanced_data['exp_coverage'] = np.exp(-enhanced_data['coverage'])
    
    # 7. 阈值特征（捕捉不同区域的特性）
    enhanced_data['high_recall'] = (enhanced_data['recall'] > 0.95).astype(int)
    enhanced_data['high_coverage'] = (enhanced_data['coverage'] > 0.95).astype(int)
    enhanced_data['both_high'] = enhanced_data['high_recall'] * enhanced_data['high_coverage']
    
    # 多级阈值特征
    enhanced_data['very_high_recall'] = (enhanced_data['recall'] > 0.98).astype(int)
    enhanced_data['very_high_coverage'] = (enhanced_data['coverage'] > 0.98).astype(int)
    enhanced_data['medium_recall'] = ((enhanced_data['recall'] > 0.9) & (enhanced_data['recall'] <= 0.95)).astype(int)
    enhanced_data['medium_coverage'] = ((enhanced_data['coverage'] > 0.9) & (enhanced_data['coverage'] <= 0.95)).astype(int)
    
    # 8. 差值特征
    enhanced_data['recall_coverage_diff'] = enhanced_data['recall'] - enhanced_data['coverage']
    enhanced_data['abs_recall_coverage_diff'] = np.abs(enhanced_data['recall_coverage_diff'])
    
    # 9. 复合特征
    enhanced_data['harmonic_mean'] = 2 * enhanced_data['recall'] * enhanced_data['coverage'] / (enhanced_data['recall'] + enhanced_data['coverage'] + 1e-10)
    enhanced_data['geometric_mean'] = np.sqrt(enhanced_data['recall'] * enhanced_data['coverage'])
    
    # F-score相关特征（不同beta值）
    enhanced_data['f1_score'] = enhanced_data['harmonic_mean']  # F1就是harmonic mean
    enhanced_data['f2_score'] = (5 * enhanced_data['recall'] * enhanced_data['coverage']) / (4 * enhanced_data['recall'] + enhanced_data['coverage'] + 1e-10)
    enhanced_data['f0_5_score'] = (1.25 * enhanced_data['recall'] * enhanced_data['coverage']) / (0.25 * enhanced_data['recall'] + enhanced_data['coverage'] + 1e-10)
    
    # 10. 分段线性特征（模拟分段函数）
    enhanced_data['recall_high_region'] = np.maximum(0, enhanced_data['recall'] - 0.9)
    enhanced_data['coverage_high_region'] = np.maximum(0, enhanced_data['coverage'] - 0.9)
    
    # 极值区域特征
    enhanced_data['recall_extreme_high'] = np.maximum(0, enhanced_data['recall'] - 0.95)
    enhanced_data['coverage_extreme_high'] = np.maximum(0, enhanced_data['coverage'] - 0.95)
    
    print(f"特征工程完成，从 {len(data.columns)} 个特征扩展到 {len(enhanced_data.columns)} 个特征")
    
    # 显示新增的特征列表
    original_cols = set(data.columns)
    new_cols = [col for col in enhanced_data.columns if col not in original_cols]
    print(f"新增特征: {new_cols}")
    
    return enhanced_data


def get_feature_columns(exclude_cols=['error', 'is_test', 'filter_id']):
    """
    获取用于训练的特征列名
    
    Args:
        exclude_cols: 需要排除的列名列表
    
    Returns:
        特征列名列表
    """
    # 基础特征
    feature_cols = ['recall', 'coverage']
    
    # 增强特征
    enhanced_features = [
        # 交互特征
        'recall_coverage_product', 'recall_coverage_ratio', 'coverage_recall_ratio',
        # 多项式特征
        'recall_squared', 'coverage_squared', 'recall_cubed', 'coverage_cubed',
        # 对数特征
        'log_recall', 'log_coverage', 'log_product',
        # 距离特征
        'distance_to_perfect', 'distance_to_origin', 'manhattan_distance_to_perfect',
        # 关系特征（替代三角函数）
        'harmony_score', 'balance_score', 'recall_rank', 'coverage_rank',
        'rank_correlation', 'rank_difference',
        # 指数特征
        'exp_recall', 'exp_coverage',
        # 阈值特征
        'high_recall', 'high_coverage', 'both_high',
        'very_high_recall', 'very_high_coverage', 'medium_recall', 'medium_coverage',
        # 差值特征
        'recall_coverage_diff', 'abs_recall_coverage_diff',
        # 复合特征
        'harmonic_mean', 'geometric_mean',
        'f1_score', 'f2_score', 'f0_5_score',
        # 分段特征
        'recall_high_region', 'coverage_high_region',
        'recall_extreme_high', 'coverage_extreme_high'
    ]
    
    return feature_cols + enhanced_features


def train_evaluate_model_enhanced(data_path, output_dir, learning_rate=0.03, monotone_constraints=None, 
                                use_quantile_regression=False, quantile_alpha=0.95, min_coverage=None, 
                                use_feature_engineering=True):
    """
    使用增强特征训练和评估LightGBM模型
    
    Args:
        data_path: CSV文件路径，包含recall,coverage,error,is_test列
        output_dir: 输出目录，保存模型和评估结果
        learning_rate: 学习率
        monotone_constraints: 单调性约束 [recall_constraint, coverage_constraint]
        use_quantile_regression: 是否使用分位数回归
        quantile_alpha: 分位数水平
        min_coverage: 最小覆盖率阈值
        use_feature_engineering: 是否使用特征工程
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 从CSV文件加载数据
    data = pd.read_csv(data_path)
    print(f"原始数据加载完成: {len(data)} 个样本")
    
    # 过滤低coverage数据
    if min_coverage is not None:
        original_count = len(data)
        data = data[data['coverage'] >= min_coverage]
        filtered_count = original_count - len(data)
        print(f"过滤掉 {filtered_count} 个低覆盖率(<{min_coverage})样本，剩余 {len(data)} 个样本")
    
    # 应用特征工程
    if use_feature_engineering:
        data = enhance_features(data)
        feature_columns = get_feature_columns()
    else:
        feature_columns = ['recall', 'coverage']
        print("未使用特征工程，仅使用原始特征")
    
    # 检查filter_id
    filter_id = None
    if 'filter_id' in data.columns:
        filter_ids = data['filter_id'].unique()
        if len(filter_ids) == 1:
            filter_id = filter_ids[0]
            print(f"数据来自单个过滤器 (ID: {filter_id})")
        else:
            print(f"数据来自多个过滤器 (共 {len(filter_ids)} 个)")
    
    # 分离训练和测试数据
    train_data = data[data['is_test'] == 0]
    test_data = data[data['is_test'] == 1]
    
    # 检查数据量
    print(f"训练集样本数: {len(train_data)}")
    print(f"测试集样本数: {len(test_data)}")
    print(f"使用特征数量: {len(feature_columns)}")
    
    if len(train_data) < 5 or len(test_data) < 2:
        print("警告: 数据量过少，可能影响模型质量")
    
    # 显示数据分布信息
    print(f"Recall范围: {train_data['recall'].min():.3f} - {train_data['recall'].max():.3f}")
    print(f"Coverage范围: {train_data['coverage'].min():.3f} - {train_data['coverage'].max():.3f}")
    print(f"Error范围: {train_data['error'].min():.3f} - {train_data['error'].max():.3f}")
    
    # 准备训练数据
    X_train = train_data[feature_columns]
    y_train = train_data['error']
    
    # 准备测试数据
    X_test = test_data[feature_columns]
    y_test = test_data['error']
    
    # 调整单调性约束以适应新特征（仅对前两个基础特征应用约束）
    if monotone_constraints is None:
        monotone_constraints = [0, 0]  # 基础特征的约束
    
    # 为所有特征设置约束（新特征默认无约束）
    full_monotone_constraints = monotone_constraints + [0] * (len(feature_columns) - 2)
    
    # 根据是否使用分位数回归设置不同的参数
    if use_quantile_regression:
        # 分位数回归参数
        params = {
            'objective': 'quantile',
            'alpha': quantile_alpha,
            'metric': 'quantile',
            'learning_rate': learning_rate,
            'max_depth': 8,  # 增加深度以处理更多特征
            'num_leaves': 31,  # 增加叶子数
            'min_data_in_leaf': 5,
            'lambda_l1': 0.01,
            'lambda_l2': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,  # 添加bagging
            'bagging_freq': 1,
            'verbose': -1
        }
        print(f"开始训练 LightGBM 分位数回归模型 (quantile) - 使用 {len(feature_columns)} 个特征")
        print(f"分位数水平: {quantile_alpha} ({quantile_alpha*100:.1f}%分位数)")
        
    else:
        # 普通回归参数
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': learning_rate,
            'max_depth': 8,
            'num_leaves': 31,
            'min_data_in_leaf': 5,
            'lambda_l1': 0.01,
            'lambda_l2': 0.1,
            'monotone_constraints': full_monotone_constraints,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'verbose': -1
        }
        print(f"开始训练 LightGBM 单调性回归模型 - 使用 {len(feature_columns)} 个特征")
        print(f"单调性约束: recall={monotone_constraints[0]}, coverage={monotone_constraints[1]}")
        print("(1: 非递减, -1: 非递增, 0: 无约束)")
    
    # 训练模型
    callbacks = []
    valid_sets = None
    
    if len(test_data) > 0:
        valid_sets = [lgb.Dataset(X_test, y_test)]
        callbacks = [lgb.early_stopping(50)]  # 增加early_stopping轮数
    
    gbm = lgb.train(params,
                    lgb.Dataset(X_train, y_train),
                    num_boost_round=200,  # 增加训练轮数
                    valid_sets=valid_sets,
                    callbacks=callbacks)
    
    # 输出特征重要性
    feature_importance = gbm.feature_importance(importance_type='gain')
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print("\n特征重要性排名 (Top 10):")
    print(importance_df.head(10).to_string(index=False))
    
    # 保存特征重要性
    importance_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"特征重要性已保存到: {importance_path}")
    
    # 保存模型
    if use_quantile_regression:
        if use_feature_engineering:
            constraint_str = f"quantile_{quantile_alpha}_enhanced"
        else:
            constraint_str = f"quantile_{quantile_alpha}"
    else:
        if use_feature_engineering:
            constraint_str = f"mono_{monotone_constraints[0]}_{monotone_constraints[1]}_enhanced"
        else:
            constraint_str = f"mono_{monotone_constraints[0]}_{monotone_constraints[1]}"
    
    model_path = os.path.join(output_dir, f"model_{constraint_str}.txt")
    gbm.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 在训练集上进行评估
    y_train_pred = gbm.predict(X_train)
    
    # 计算训练集性能指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    
    # 计算训练集安全预测比例 (predicted >= actual)
    train_diffs = y_train_pred - y_train
    train_safe = np.sum(train_diffs >= 0)
    train_safe_ratio = train_safe / len(y_train)
    
    # 检查预测值的唯一性
    unique_pred = len(np.unique(y_train_pred))
    print(f"训练集预测值唯一数量: {unique_pred} / {len(y_train_pred)} ({unique_pred/len(y_train_pred)*100:.1f}%)")
    
    # 如果是分位数回归，还要计算分位数损失和覆盖率
    if use_quantile_regression:
        # 计算分位数损失
        train_quantile_loss = quantile_loss_function(y_train, y_train_pred, quantile_alpha)
        
        # 计算实际覆盖率 (应该接近quantile_alpha)
        train_coverage = np.mean(y_train <= y_train_pred)
        
        print(f"\n训练集评估结果 (分位数回归):")
        print(f"MSE: {train_mse:.6f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}")
        print(f"分位数损失: {train_quantile_loss:.6f}")
        print(f"安全预测比例 (predicted >= actual): {train_safe_ratio:.2%}")
        print(f"实际覆盖率: {train_coverage:.3f} (期望: {quantile_alpha:.3f})")
    else:
        print(f"\n训练集评估结果:")
        print(f"MSE: {train_mse:.6f}, RMSE: {train_rmse:.6f}, MAE: {train_mae:.6f}")
        print(f"安全预测比例 (predicted >= actual): {train_safe_ratio:.2%}")
    
    # 如果有测试集，在测试集上进行评估
    test_results = {}
    if len(test_data) > 0:
        y_test_pred = gbm.predict(X_test)
        
        # 计算测试集性能指标
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # 计算测试集安全预测比例
        test_diffs = y_test_pred - y_test
        test_safe = np.sum(test_diffs >= 0)
        test_safe_ratio = test_safe / len(y_test)
        
        # 检查测试集预测值的唯一性
        unique_pred_test = len(np.unique(y_test_pred))
        print(f"测试集预测值唯一数量: {unique_pred_test} / {len(y_test_pred)} ({unique_pred_test/len(y_test_pred)*100:.1f}%)")
        
        test_results = {
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_safe_ratio': test_safe_ratio
        }
        
        if use_quantile_regression:
            # 计算测试集分位数损失和覆盖率
            test_quantile_loss = quantile_loss_function(y_test, y_test_pred, quantile_alpha)
            test_coverage = np.mean(y_test <= y_test_pred)
            
            test_results.update({
                'test_quantile_loss': test_quantile_loss,
                'test_coverage': test_coverage
            })
            
            print(f"\n测试集评估结果 (分位数回归):")
            print(f"MSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")
            print(f"分位数损失: {test_quantile_loss:.6f}")
            print(f"安全预测比例 (predicted >= actual): {test_safe_ratio:.2%}")
            print(f"实际覆盖率: {test_coverage:.3f} (期望: {quantile_alpha:.3f})")
        else:
            print(f"\n测试集评估结果:")
            print(f"MSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}, MAE: {test_mae:.6f}")
            print(f"安全预测比例 (predicted >= actual): {test_safe_ratio:.2%}")
    
    return {
        'model': gbm,
        'train_mse': train_mse,
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_safe_ratio': train_safe_ratio,
        'use_quantile_regression': use_quantile_regression,
        'quantile_alpha': quantile_alpha if use_quantile_regression else None,
        **test_results
    }


def quantile_loss_function(y_true, y_pred, quantile):
    """
    计算分位数损失函数
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def train_all_filters_individually_enhanced(data_dir, output_dir, learning_rate=0.03, monotone_constraints=None, 
                                          min_coverage=None, use_quantile_regression=False, quantile_alpha=0.85, 
                                          max_workers=None, use_feature_engineering=True):
    """为每个滤波器分别训练增强模型"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 寻找所有滤波器的数据文件
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('filter_') and (f.endswith('_triples.csv') or f.endswith('_triples_augmented.csv') or f.endswith('_triples_augmented_augmented.csv'))]
    
    if not csv_files:
        print(f"错误: 在 {data_dir} 中未找到任何滤波器数据文件")
        return
    
    feature_type = "增强特征" if use_feature_engineering else "基础特征"
    print(f"找到 {len(csv_files)} 个滤波器数据文件，使用{feature_type}")
    
    # 设置并行进程数
    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(csv_files))
    
    print(f"使用 {max_workers} 个并行进程进行训练")
    
    # 准备参数列表
    args_list = []
    for csv_file in csv_files:
        filter_id = csv_file.split('_')[1]  # 提取滤波器ID
        data_path = os.path.join(data_dir, csv_file)
        filter_output_dir = os.path.join(output_dir, f"filter_{filter_id}")
        
        args_list.append((
            data_path, 
            filter_output_dir, 
            learning_rate, 
            monotone_constraints, 
            min_coverage,
            use_quantile_regression,
            quantile_alpha,
            filter_id,
            use_feature_engineering
        ))
    
    # 存储所有滤波器的评估结果
    all_results = []
    failed_filters = []
    
    # 使用进程池并行训练
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_filter = {executor.submit(train_single_filter_enhanced, args): args[-2] for args in args_list}  # args[-2] 是 filter_id
        
        # 收集结果
        completed = 0
        for future in as_completed(future_to_filter):
            filter_id = future_to_filter[future]
            completed += 1
            
            try:
                result = future.result()
                if result['success']:
                    all_results.append(result)
                    print(f"[{completed}/{len(csv_files)}] 滤波器 {filter_id} 训练完成")
                else:
                    failed_filters.append(filter_id)
                    print(f"[{completed}/{len(csv_files)}] 滤波器 {filter_id} 训练失败: {result.get('error', '未知错误')}")
                    
            except Exception as exc:
                failed_filters.append(filter_id)
                print(f"[{completed}/{len(csv_files)}] 滤波器 {filter_id} 训练异常: {exc}")
    
    # 报告结果
    print(f"\n训练完成！成功: {len(all_results)}, 失败: {len(failed_filters)}")
    if failed_filters:
        print(f"失败的滤波器: {failed_filters}")
    
    if not all_results:
        print("没有成功训练的模型，无法计算平均指标")
        return None
    
    # 计算平均指标
    train_mse_values = [r['train_mse'] for r in all_results]
    train_rmse_values = [r['train_rmse'] for r in all_results]
    train_mae_values = [r['train_mae'] for r in all_results]
    train_safe_ratios = [r['train_safe_ratio'] for r in all_results]
    
    test_mse_values = [r['test_mse'] for r in all_results if r['test_mse'] is not None]
    test_rmse_values = [r['test_rmse'] for r in all_results if r['test_rmse'] is not None]
    test_mae_values = [r['test_mae'] for r in all_results if r['test_mae'] is not None]
    test_safe_ratios = [r['test_safe_ratio'] for r in all_results if r['test_safe_ratio'] is not None]
    
    avg_train_mse = np.mean(train_mse_values)
    avg_train_rmse = np.mean(train_rmse_values)
    avg_train_mae = np.mean(train_mae_values)
    avg_train_safe_ratio = np.mean(train_safe_ratios)
    
    avg_test_mse = np.mean(test_mse_values) if test_mse_values else None
    avg_test_rmse = np.mean(test_rmse_values) if test_rmse_values else None
    avg_test_mae = np.mean(test_mae_values) if test_mae_values else None
    avg_test_safe_ratio = np.mean(test_safe_ratios) if test_safe_ratios else None
    
    # 输出平均指标
    print(f"\n所有滤波器的平均评估指标 ({feature_type}):")
    print("==================================================")
    print(f"训练集平均MSE: {avg_train_mse:.6f}")
    print(f"训练集平均RMSE: {avg_train_rmse:.6f}")
    print(f"训练集平均MAE: {avg_train_mae:.6f}")
    print(f"训练集平均安全预测比例: {avg_train_safe_ratio:.2%}")
    
    if avg_test_mse is not None:
        print(f"\n测试集平均MSE: {avg_test_mse:.6f}")
        print(f"测试集平均RMSE: {avg_test_rmse:.6f}")
        print(f"测试集平均MAE: {avg_test_mae:.6f}")
        print(f"测试集平均安全预测比例: {avg_test_safe_ratio:.2%}")

    # 返回平均指标
    return {
        'avg_train_mse': avg_train_mse,
        'avg_train_rmse': avg_train_rmse,
        'avg_train_mae': avg_train_mae,
        'avg_train_safe_ratio': avg_train_safe_ratio,
        'avg_test_mse': avg_test_mse,
        'avg_test_rmse': avg_test_rmse,
        'avg_test_mae': avg_test_mae,
        'avg_test_safe_ratio': avg_test_safe_ratio,
        'successful_filters': len(all_results),
        'failed_filters': len(failed_filters)
    }


def train_single_filter_enhanced(args_tuple):
    """训练单个滤波器的包装函数，用于多进程"""
    data_path, filter_output_dir, learning_rate, monotone_constraints, min_coverage, use_quantile_regression, quantile_alpha, filter_id, use_feature_engineering = args_tuple
    
    try:
        feature_type = "增强特征" if use_feature_engineering else "基础特征"
        print(f"开始训练滤波器 {filter_id} ({feature_type})")
        result = train_evaluate_model_enhanced(
            data_path,
            filter_output_dir,
            learning_rate=learning_rate,
            monotone_constraints=monotone_constraints,
            min_coverage=min_coverage,
            use_quantile_regression=use_quantile_regression,
            quantile_alpha=quantile_alpha,
            use_feature_engineering=use_feature_engineering
        )
        
        print(f"完成训练滤波器 {filter_id}")
        return {
            'filter_id': filter_id,
            'train_mse': result['train_mse'],
            'train_rmse': result['train_rmse'],
            'train_mae': result['train_mae'],
            'train_safe_ratio': result['train_safe_ratio'],
            'test_mse': result.get('test_mse'),
            'test_rmse': result.get('test_rmse'),
            'test_mae': result.get('test_mae'),
            'test_safe_ratio': result.get('test_safe_ratio'),
            'success': True
        }
        
    except Exception as e:
        print(f"训练滤波器 {filter_id} 时出错: {str(e)}")
        return {
            'filter_id': filter_id,
            'error': str(e),
            'success': False
        }


def main():
    """主函数，解析命令行参数并执行训练评估"""
    parser = argparse.ArgumentParser(description='使用LightGBM增强特征训练和评估(recall, coverage) -> error回归模型')
    parser.add_argument('--data', type=str, help='CSV文件路径，包含recall,coverage,error,is_test列')
    parser.add_argument('--output', type=str, help='输出目录，用于保存模型和评估结果')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='学习率')
    parser.add_argument('--monotone_constraints', type=int, nargs=2, default=[0, 0], 
                        help='单调性约束 [recall_constraint, coverage_constraint] (1: 非递减, -1: 非递增, 0: 无约束)')
    parser.add_argument('--quantile', action='store_true', help='使用分位数回归')
    parser.add_argument('--quantile_alpha', type=float, default=0.95, help='分位数水平 (默认0.95)')
    parser.add_argument('--data_dir', type=str, help='数据目录，包含多个CSV文件')
    parser.add_argument('--individual', action='store_true', help='为每个滤波器单独训练模型')
    parser.add_argument('--min_coverage', type=float, default=0.8, help='最小覆盖率阈值，低于此值的样本将被过滤')
    parser.add_argument('--max_workers', type=int, default=None, help='最大并行进程数，默认使用CPU核心数')
    parser.add_argument('--no_feature_engineering', action='store_true', help='禁用特征工程，仅使用基础特征 (recall, coverage)')
    args = parser.parse_args()
    
    # 确定是否使用特征工程
    use_feature_engineering = not args.no_feature_engineering
    feature_type = "增强特征" if use_feature_engineering else "基础特征"
    
    # 确定数据路径和输出目录
    if args.individual:
        # 个别模型模式
        if not args.data_dir:
            print("错误: 使用 --individual 选项时必须指定 --data_dir")
            return 1
        
        # 规范化数据目录路径
        data_dir = os.path.abspath(args.data_dir)
        
        # 处理输出目录：优先使用用户指定的路径，否则使用默认路径
        if args.output:
            output_dir = os.path.abspath(args.output)
        else:
            # 默认在数据目录的上级目录创建输出目录
            output_dir = os.path.join(os.path.dirname(data_dir), 'lgbm_enhanced_models')
            output_dir = os.path.abspath(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"数据目录: {data_dir}")
        print(f"输出目录: {output_dir}")
        print(f"特征类型: {feature_type}")
        print(f"单调性约束: recall={args.monotone_constraints[0]}, coverage={args.monotone_constraints[1]}")
        
        # 为每个滤波器单独训练模型
        train_all_filters_individually_enhanced(
            data_dir,
            output_dir,
            learning_rate=args.learning_rate,
            monotone_constraints=args.monotone_constraints,
            min_coverage=args.min_coverage,
            use_quantile_regression=args.quantile,
            quantile_alpha=args.quantile_alpha,
            max_workers=args.max_workers,
            use_feature_engineering=use_feature_engineering
        )

    else:
        # 单一模型训练
        if not args.data:
            print("错误: 训练单一模型时必须指定 --data")
            return 1
        
        # 规范化数据文件路径
        data_path = os.path.abspath(args.data)
        if not os.path.exists(data_path):
            print(f"错误: 数据文件不存在: {data_path}")
            return 1
        
        # 处理输出目录：优先使用用户指定的路径，否则使用默认路径
        if args.output:
            output_dir = os.path.abspath(args.output)
        else:
            # 默认在数据文件同目录下创建输出目录
            output_dir = os.path.join(os.path.dirname(data_path), 'lgbm_enhanced_model')
            output_dir = os.path.abspath(output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"数据文件: {data_path}")
        print(f"输出目录: {output_dir}")
        print(f"特征类型: {feature_type}")
        print(f"单调性约束: recall={args.monotone_constraints[0]}, coverage={args.monotone_constraints[1]}")
        
        # 训练单一模型
        train_evaluate_model_enhanced(
            data_path,
            output_dir,
            learning_rate=args.learning_rate,
            monotone_constraints=args.monotone_constraints,
            use_quantile_regression=args.quantile,
            quantile_alpha=args.quantile_alpha,
            min_coverage=args.min_coverage,
            use_feature_engineering=use_feature_engineering
        )
    
    return 0


if __name__ == "__main__":
    main()


# 使用示例：
# 单一模型训练（使用增强特征）
# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm_enhanced.py --data /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented/filter_58_triples_augmented.csv --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_enhanced --quantile --quantile_alpha 0.99 --min_coverage 0.8

# 批量训练（使用增强特征）
# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm_enhanced.py --data_dir /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented --individual --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/enhanced_models --quantile --quantile_alpha 0.95 --min_coverage 0.8 --max_workers 10

# 对比训练（不使用增强特征）
# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm_enhanced.py --data /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented/filter_58_triples_augmented.csv --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/basic_models --quantile --quantile_alpha 0.95 --min_coverage 0.8 --no_feature_engineering