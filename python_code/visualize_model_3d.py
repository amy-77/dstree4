#!/usr/bin/env python3
"""
可视化LightGBM模型的3D形状和原始数据点
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')  # 设置为非交互式后端
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize


def load_model_and_data(model_path, data_path):
    """加载模型和数据"""
    # 加载模型
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = lgb.Booster(model_file=model_path)
    print(f"✓ 模型已加载: {model_path}")
    
    # 加载数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = pd.read_csv(data_path)
    print(f"✓ 数据已加载: {data_path}")
    print(f"  数据点数量: {len(data)}")
    print(f"  Recall范围: [{data['recall'].min():.3f}, {data['recall'].max():.3f}]")
    print(f"  Coverage范围: [{data['coverage'].min():.3f}, {data['coverage'].max():.3f}]")
    print(f"  Error范围: [{data['error'].min():.6f}, {data['error'].max():.6f}]")
    
    return model, data


def create_prediction_surface(model, recall_range, coverage_range, resolution=50):
    """创建预测曲面"""
    # 创建网格
    recall_grid = np.linspace(recall_range[0], recall_range[1], resolution)
    coverage_grid = np.linspace(coverage_range[0], coverage_range[1], resolution)
    R, C = np.meshgrid(recall_grid, coverage_grid)
    
    # 预测整个网格
    grid_points = np.column_stack([R.ravel(), C.ravel()])
    predictions = model.predict(grid_points)
    Z = predictions.reshape(R.shape)
    
    return R, C, Z


def plot_3d_model_and_data(model, data, filter_id, output_dir=None):
    """绘制3D模型曲面和原始数据点"""
    
    # 过滤数据：只显示coverage > 0.8的数据点
    filtered_data = data[data['coverage'] > 0.8].copy()
    print(f"过滤后数据点数量: {len(filtered_data)} (coverage > 0.8)")
    
    if len(filtered_data) == 0:
        print("警告: 没有coverage > 0.8的数据点")
        return None, None
    
    # 按coverage降序，然后按recall降序排序，让高coverage和高recall的点先绘制
    filtered_data = filtered_data.sort_values(['coverage', 'recall'], ascending=[False, False])
    print(f"数据已按coverage和recall降序排序")
    
    # 固定绘图范围
    recall_range = (0.95, 1.0)
    coverage_range = (0.8, 1.0)
    
    print(f"绘图范围:")
    print(f"  Recall: [{recall_range[0]:.3f}, {recall_range[1]:.3f}]")
    print(f"  Coverage: [{coverage_range[0]:.3f}, {coverage_range[1]:.3f}]")
    
    # 创建预测曲面
    print("正在生成预测曲面...")
    R, C, Z = create_prediction_surface(model, recall_range, coverage_range, resolution=50)
    
    # 创建图形 - 只显示两个子图
    fig = plt.figure(figsize=(16, 7))
    
    # 第一个图：原始数据散点图
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 绘制原始数据点，使用较大的点和渐变透明度
    # 根据error值确定点的大小
    point_sizes = 20 + (filtered_data['error'] - filtered_data['error'].min()) / (filtered_data['error'].max() - filtered_data['error'].min()) * 30
    
    scatter = ax1.scatter(filtered_data['recall'], filtered_data['coverage'], filtered_data['error'], 
                         c=filtered_data['error'], cmap='plasma', s=point_sizes, alpha=0.7, 
                         edgecolors='black', linewidth=0.3)
    
    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Coverage', fontsize=12)
    ax1.set_zlabel('Error', fontsize=12)
    ax1.set_title(f'Filter {filter_id}: Original Training Data\n(Coverage > 0.8, sorted by Cov↓, Rec↓)', fontsize=14)
    ax1.set_xlim(recall_range)
    ax1.set_ylim(coverage_range)
    
    # 添加颜色条
    cbar1 = plt.colorbar(scatter, ax=ax1, shrink=0.6, aspect=20)
    cbar1.set_label('Actual Error', rotation=270, labelpad=15)
    
    # 第二个图：模型预测曲面
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 绘制预测曲面，增加透明度
    surface = ax2.plot_surface(R, C, Z, cmap='viridis', alpha=0.5, edgecolor='none')
    
    # 叠加原始数据点，使用不同的颜色映射和大小
    scatter2 = ax2.scatter(filtered_data['recall'], filtered_data['coverage'], filtered_data['error'], 
                          c=filtered_data['coverage'], cmap='coolwarm', s=point_sizes, alpha=0.8, 
                          edgecolors='white', linewidth=0.5)
    
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Coverage', fontsize=12)
    ax2.set_zlabel('Error', fontsize=12)
    ax2.set_title(f'Filter {filter_id}: Model Prediction Surface\nwith Training Data (colored by Coverage)', fontsize=14)
    ax2.set_xlim(recall_range)
    ax2.set_ylim(coverage_range)
    
    # 添加颜色条
    cbar2 = plt.colorbar(surface, ax=ax2, shrink=0.6, aspect=20)
    cbar2.set_label('Predicted Error', rotation=270, labelpad=15)
    
    # 调整视角，使用更好的观察角度
    ax1.view_init(elev=25, azim=35)
    ax2.view_init(elev=25, azim=35)
    
    plt.tight_layout()
    
    # 保存图像
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"filter_{filter_id}_3d_simplified_sorted.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 简化3D图已保存: {output_path}")
    
    # 关闭图形以释放内存
    plt.close(fig)
    
    # 创建单独的高质量模型图，使用更好的排序和颜色
    fig2 = plt.figure(figsize=(12, 9))
    ax_3d = fig2.add_subplot(111, projection='3d')
    
    # 绘制预测曲面
    surface = ax_3d.plot_surface(R, C, Z, cmap='viridis', alpha=0.4, edgecolor='none')
    
    # 创建分层的数据点可视化
    # 按recall分组，用不同透明度绘制
    recall_bins = np.linspace(filtered_data['recall'].min(), filtered_data['recall'].max(), 6)
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'blue', 'purple']
    
    for i in range(len(recall_bins)-1):
        mask = (filtered_data['recall'] >= recall_bins[i]) & (filtered_data['recall'] < recall_bins[i+1])
        if mask.sum() > 0:
            subset = filtered_data[mask]
            ax_3d.scatter(subset['recall'], subset['coverage'], subset['error'], 
                         c=colors[i], s=60, alpha=0.8, 
                         edgecolors='black', linewidth=0.3,
                         label=f'Recall [{recall_bins[i]:.3f}, {recall_bins[i+1]:.3f})')
    
    ax_3d.set_xlabel('Recall', fontsize=14, labelpad=10)
    ax_3d.set_ylabel('Coverage', fontsize=14, labelpad=10)
    ax_3d.set_zlabel('Error', fontsize=14, labelpad=10)
    ax_3d.set_title(f'Filter {filter_id}: LightGBM Model 3D Visualization\n'
                   f'Surface: Model Predictions | Points: Training Data (layered by Recall)', fontsize=16)
    ax_3d.set_xlim(recall_range)
    ax_3d.set_ylim(coverage_range)
    
    # 添加图例
    ax_3d.legend(loc='upper left', bbox_to_anchor=(0, 1), fontsize=10)
    
    # 添加颜色条
    cbar = plt.colorbar(surface, ax=ax_3d, shrink=0.6, aspect=30, pad=0.1)
    cbar.set_label('Predicted Error', rotation=270, labelpad=20, fontsize=12)
    
    # 调整视角
    ax_3d.view_init(elev=25, azim=35)
    
    if output_dir:
        output_path_3d = os.path.join(output_dir, f"filter_{filter_id}_3d_model_layered.png")
        plt.savefig(output_path_3d, dpi=300, bbox_inches='tight')
        print(f"✓ 分层3D图已保存: {output_path_3d}")
    
    # 关闭图形以释放内存
    plt.close(fig2)
    
    return fig, fig2


def analyze_model_statistics(model, data):
    """分析模型统计信息"""
    print("\n" + "="*60)
    print("模型和数据统计分析")
    print("="*60)
    
    # 对训练数据进行预测
    X_train = data[['recall', 'coverage']].values
    y_train = data['error'].values
    y_pred = model.predict(X_train)
    
    # 计算误差统计
    mae = np.mean(np.abs(y_pred - y_train))
    rmse = np.sqrt(np.mean((y_pred - y_train) ** 2))
    r2 = 1 - np.sum((y_train - y_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)
    
    print(f"训练数据预测统计:")
    print(f"  MAE (平均绝对误差): {mae:.6f}")
    print(f"  RMSE (均方根误差): {rmse:.6f}")
    print(f"  R² (决定系数): {r2:.6f}")
    
    # 预测范围 vs 数据范围
    print(f"\n预测值 vs 实际值:")
    print(f"  实际值范围: [{y_train.min():.6f}, {y_train.max():.6f}]")
    print(f"  预测值范围: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"  实际值均值: {np.mean(y_train):.6f}")
    print(f"  预测值均值: {np.mean(y_pred):.6f}")
    
    # 安全预测比例
    safe_ratio = np.mean(y_pred >= y_train)
    print(f"  安全预测比例 (pred >= actual): {safe_ratio:.2%}")


def main():
    """主函数"""
    # 配置参数
    filter_id = 190
    model_path = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/quantile_models_individual/filter_190/model_quantile_0.9.txt"
    data_path = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/lgbm_data/filter_190_triples.csv"
    output_dir = "/home/qwang/projects/leafi/dstree2/result/train_25M_R90_C90_K1_QN2/model_visualizations"
    
    print(f"正在可视化 Filter {filter_id} 的模型...")
    print(f"模型路径: {model_path}")
    print(f"数据路径: {data_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 加载模型和数据
        model, data = load_model_and_data(model_path, data_path)
        
        # 分析统计信息
        analyze_model_statistics(model, data)
        
        # 绘制3D可视化
        print("\n正在生成3D可视化...")
        fig1, fig2 = plot_3d_model_and_data(model, data, filter_id, output_dir)
        
        print(f"\n✓ Filter {filter_id} 可视化完成！")
        
    except Exception as e:
        print(f"✗ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 