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


def train_evaluate_model(data_path, output_dir, learning_rate=0.03, plot_results=False, augment_data=False, interpolate_data=False, monotone_constraints=None, use_quantile_regression=False, quantile_alpha=0.95, min_coverage=None):
    """
    Train and evaluate a LightGBM model on the given data with monotonic constraints.
    Args:
        data_path: Path to the CSV file containing recall, coverage, error triples
        output_dir: Directory to save model and evaluation results
        learning_rate: Learning rate for the model
        plot_results: Whether to generate plots of the results
        augment_data: Whether to augment high recall data points
        interpolate_data: Whether to use interpolation to augment data
        monotone_constraints: List of monotonicity constraints for features [recall_constraint, coverage_constraint]
                             1: increasing, -1: decreasing, 0: no constraint
        use_quantile_regression: Whether to use quantile regression (quantile_l2)
        quantile_alpha: Quantile level for quantile regression (e.g., 0.95 for 95th percentile)
        min_coverage: Minimum coverage threshold, samples below this will be filtered out
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 从CSV文件加载数据
    data = pd.read_csv(data_path)
    print(f"原始数据加载完成: {len(data)} 个样本")
    
    # 数据增强功能已禁用
    if interpolate_data or augment_data:
        print("警告: 数据增强和插值功能已被禁用，使用原始数据进行训练")
    
    # 过滤低coverage数据
    if min_coverage is not None:
        original_count = len(data)
        data = data[data['coverage'] >= min_coverage]
        filtered_count = original_count - len(data)
        print(f"过滤掉 {filtered_count} 个低覆盖率(<{min_coverage})样本，剩余 {len(data)} 个样本")
    
    # 如果CSV包含filter_id列，获取唯一的filter_id值
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
    
    if len(train_data) < 5 or len(test_data) < 2:
        print("警告: 数据量过少，可能影响模型质量")
    
    # 显示数据分布信息
    print(f"Recall范围: {train_data['recall'].min():.3f} - {train_data['recall'].max():.3f}")
    print(f"Coverage范围: {train_data['coverage'].min():.3f} - {train_data['coverage'].max():.3f}")
    print(f"Error范围: {train_data['error'].min():.3f} - {train_data['error'].max():.3f}")
    
    # 准备训练数据
    X_train = train_data[['recall', 'coverage']]
    y_train = train_data['error']
    
    # 准备测试数据
    X_test = test_data[['recall', 'coverage']]
    y_test = test_data['error']
    
    # 设置默认单调性约束：recall和coverage都是非递减的（error随着recall/coverage增加而增加或保持不变）
    if monotone_constraints is None:
        monotone_constraints = [0, 0]  # [recall_constraint, coverage_constraint]: 1表示非递减
    
    # 根据是否使用分位数回归设置不同的参数
    if use_quantile_regression:
        # 分位数回归参数（不使用单调性约束）
        params = {
            'objective': 'quantile',              # 分位数回归
            'alpha': quantile_alpha,              # 分位数水平 (0.95 = 95%分位数)
            'metric': 'quantile',                 # 分位数损失作为评估指标
            'learning_rate': learning_rate,
            'max_depth': 6,
            'num_leaves': 10,
            'min_data_in_leaf': 5,
            'lambda_l1': 0.01,                    # L1正则化
            'lambda_l2': 0.1,                     # L2正则化
            'feature_fraction': 0.8,
            'verbose': -1
        }
        print(f"开始训练 LightGBM 分位数回归模型 (quantile)")
        print(f"分位数水平: {quantile_alpha} ({quantile_alpha*100:.1f}%分位数)")
        print(f"单调性约束: recall={monotone_constraints[0]}, coverage={monotone_constraints[1]}")
        print("(1: 非递减, -1: 非递增, 0: 无约束)")
        
    else:
        # 普通回归 + 单调性约束参数
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'learning_rate': learning_rate,
            'max_depth': 6,
            'num_leaves':10,
            'min_data_in_leaf': 5,
            'lambda_l1': 0.01,                    # 添加L1正则化
            'lambda_l2': 0.1,                     # 添加L2正则化
            'monotone_constraints': monotone_constraints,  # 单调性约束
            'feature_fraction': 0.8,
            'verbose': -1
        }
        print(f"开始训练 LightGBM 单调性回归模型")
        print(f"单调性约束: recall={monotone_constraints[0]}, coverage={monotone_constraints[1]}")
        print("(1: 非递减, -1: 非递增, 0: 无约束)")
    
    # 训练模型
    callbacks = []
    valid_sets = None
    
    if len(test_data) > 0:
        valid_sets = [lgb.Dataset(X_test, y_test)]
        callbacks = [lgb.early_stopping(30)]
    
    gbm = lgb.train(params,
                    lgb.Dataset(X_train, y_train),
                    num_boost_round=50,
                    valid_sets=valid_sets,
                    callbacks=callbacks)
    
    # 保存模型
    if use_quantile_regression:
        constraint_str = f"quantile_{quantile_alpha}"
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
        print(f"MSE: {train_mse:.3f}, RMSE: {train_rmse:.3f}, MAE: {train_mae:.3f}")
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
    参数:
    - y_true: 真实值  - y_pred: 预测值
    - quantile: 分位数水平 (0 < quantile < 1)
    返回:
    - 分位数损失
    """
    error = y_true - y_pred
    loss = np.maximum(quantile * error, (quantile - 1) * error)
    return np.mean(loss)


def train_all_filters_individually(data_dir, output_dir, learning_rate=0.03, monotone_constraints=None, min_coverage=None, 
                                 use_quantile_regression=False, quantile_alpha=0.85, max_workers=None):
    """为每个滤波器分别训练模型，并计算所有模型的平均指标（支持多进程）
    
    Args:
        data_dir: 包含所有滤波器数据文件的目录
        output_dir: 保存模型和评估结果的目录
        learning_rate: 学习率
        monotone_constraints: 单调性约束
        min_coverage: 最小覆盖率阈值，低于此值的样本将被过滤
        use_quantile_regression: 是否使用分位数回归
        quantile_alpha: 分位数水平
        max_workers: 最大并行进程数，None表示使用CPU核心数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 寻找所有滤波器的数据文件
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('filter_') and (f.endswith('_triples.csv') or f.endswith('_triples_augmented.csv') or f.endswith('_triples_augmented_augmented.csv'))]
    
    if not csv_files:
        print(f"错误: 在 {data_dir} 中未找到任何滤波器数据文件")
        return
    
    print(f"找到 {len(csv_files)} 个滤波器数据文件")
    
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
            filter_id
        ))
    
    # 存储所有滤波器的评估结果
    all_results = []
    failed_filters = []
    
    # 使用进程池并行训练
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_filter = {executor.submit(train_single_filter, args): args[-1] for args in args_list}
        
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
    print("\n所有滤波器的平均评估指标:")
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


def train_single_filter(args_tuple):
    """训练单个滤波器的包装函数，用于多进程"""
    data_path, filter_output_dir, learning_rate, monotone_constraints, min_coverage, use_quantile_regression, quantile_alpha, filter_id = args_tuple
    
    try:
        print(f"开始训练滤波器 {filter_id}")
        result = train_evaluate_model(
            data_path,
            filter_output_dir,
            learning_rate=learning_rate,
            monotone_constraints=monotone_constraints,
            min_coverage=min_coverage,
            use_quantile_regression=use_quantile_regression,
            quantile_alpha=quantile_alpha
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
    parser = argparse.ArgumentParser(description='使用LightGBM训练和评估(recall, coverage) -> error单调性回归模型')
    parser.add_argument('--data', type=str, help='CSV文件路径，包含recall,coverage,error,is_test列')
    parser.add_argument('--output', type=str, help='输出目录，用于保存模型和评估结果')
    parser.add_argument('--learning_rate', type=float, default=0.03, help='学习率')
    parser.add_argument('--monotone_constraints', type=int, nargs=2, default=[1, 1], 
                        help='单调性约束 [recall_constraint, coverage_constraint] (1: 非递减, -1: 非递增, 0: 无约束)')
    parser.add_argument('--quantile', action='store_true', help='使用分位数回归')
    parser.add_argument('--quantile_alpha', type=float, default=0.85, help='分位数水平 (默认0.85)')
    parser.add_argument('--data_dir', type=str, help='数据目录，包含多个CSV文件 (将使用all_filters_triples.csv)')
    parser.add_argument('--individual', action='store_true', help='为每个滤波器单独训练模型')
    parser.add_argument('--min_coverage', type=float, default=0.7, help='最小覆盖率阈值，低于此值的样本将被过滤')
    parser.add_argument('--max_workers', type=int, default=None, help='最大并行进程数，默认使用CPU核心数')
    args = parser.parse_args()
    
    # 确定数据路径和输出目录
    if args.individual:
        # 个别模型模式
        if not args.data_dir:
            print("错误: 使用 --individual 选项时必须指定 --data_dir")
            return 1
        
        output_dir = args.output if args.output else os.path.join(args.data_dir, '../lgbm_individual_models')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"数据目录: {args.data_dir}")
        print(f"输出目录: {output_dir}")
        print(f"单调性约束: recall={args.monotone_constraints[0]}, coverage={args.monotone_constraints[1]}")
        
        # 为每个滤波器单独训练模型
        train_all_filters_individually(
            args.data_dir,
            output_dir,
            learning_rate=args.learning_rate,
            monotone_constraints=args.monotone_constraints,
            min_coverage=args.min_coverage,
            use_quantile_regression=args.quantile,
            quantile_alpha=args.quantile_alpha,
            max_workers=args.max_workers
        )

    else:
        # 单一模型训练
        if not args.data:
            print("错误: 训练单一模型时必须指定 --data")
            return 1
        
        if not os.path.exists(args.data):
            print(f"错误: 数据文件不存在: {args.data}")
            return 1
        
        output_dir = args.output if args.output else os.path.join(os.path.dirname(args.data), 'lgbm_model')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"数据文件: {args.data}")
        print(f"输出目录: {output_dir}")
        print(f"单调性约束: recall={args.monotone_constraints[0]}, coverage={args.monotone_constraints[1]}")
        
        # 训练单一模型
        train_evaluate_model(
            args.data,
            output_dir,
            learning_rate=args.learning_rate,
            monotone_constraints=args.monotone_constraints,
            use_quantile_regression=args.quantile,
            quantile_alpha=args.quantile_alpha,
            min_coverage=args.min_coverage
        )
    
    return 0




if __name__ == "__main__":
    main() 





# 不带增强
# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm.py --data /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/filter_58_triples.csv --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models --learning_rate 0.03 --min_coverage 0.8 --quantile --quantile_alpha 0.9 

#带增强
# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm.py --data /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented/filter_58_triples_augmented_augmented.csv --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models --learning_rate 0.03 --quantile --quantile_alpha 0.9 --min_coverage 0.5  

# python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented --individual --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_individual_95 --learning_rate 0.03 --quantile --quantile_alpha 0.95 --min_coverage 0.8 --max_workers 40


# 2389428
# nohup python /home/qwang/projects/leafi/dstree2/result/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data_augmented --individual --output /home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_individual_95 --learning_rate 0.03 --quantile --quantile_alpha 0.95 --min_coverage 0.8 --max_workers 40 > lgbm_training.log 2>&1 &