//
// Created by Qitong Wang on 2023/2/20.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "conformal.h"

#include <algorithm>
#include <fstream>
#include <cmath>


// #include "interpolation.h"  // ALGLIB的二维插值头文件
#include "alglib/interpolation.h"
#include "spdlog/spdlog.h"

#include "comp.h"
#include "vec.h"
#include "config.h"
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <unsupported/Eigen/Splines>

namespace upcite {

upcite::ConformalRegressor::ConformalRegressor(std::string core_type_str,
                                               VALUE_TYPE confidence) :
    gsl_accel_(nullptr),
    gsl_spline_(nullptr){
  if (core_type_str == "discrete") {
    core_ = DISCRETE;
  } else if (core_type_str == "spline") {
    core_ = SPLINE;
  } else {
    spdlog::error("conformal core {:s} is not recognized; roll back to the default: discrete",
                  core_type_str);
    core_ = DISCRETE;
  }

  confidence_level_ = confidence;
}

RESPONSE upcite::ConformalRegressor::fit(std::vector<ERROR_TYPE> &residuals) {
  // printf("------------进入conformal_predictor_->fit(residuals)-----------\n");
  if (!alphas_.empty()) {
    spdlog::warn("conformal alphas have been set; clear");
    alphas_.clear();
  }
  // 将残差的绝对值存储到 alphas_ 中
  alphas_.assign(residuals.begin(), residuals.end());
  for (auto &alpha : alphas_) { alpha = alpha < 0 ? -alpha : alpha; }

  // 对alphas_进行排序（从小到大）
  std::sort(alphas_.begin(), alphas_.end()); // non-decreasing
  
  // 添加哨兵值
  if (!alphas_.empty()) {
    // 计算统计量
    double sum = 0.0;
    double max_val = alphas_.back();  // 排序后最大值在末尾
    
    for (const auto& alpha : alphas_) {
      sum += alpha;
    }
    double mean = sum / alphas_.size();
    
    // 计算标准差
    double sum_sq_diff = 0.0;
    for (const auto& alpha : alphas_) {
      double diff = alpha - mean;
      sum_sq_diff += diff * diff;
    }
    double std_dev = std::sqrt(sum_sq_diff / alphas_.size());
    
    // 计算哨兵值：max(最大值, mean + 3*std)
    double sentinel_value = std::max(max_val, mean + 3.0 * std_dev);
    
    // printf("添加哨兵值: max=%.6f, mean=%.6f, std=%.6f, mean+3*std=%.6f, 选择哨兵值=%.6f\n",
    //        max_val, mean, std_dev, mean + 3.0 * std_dev, sentinel_value);
    
    // 在开头插入0
    alphas_.insert(alphas_.begin(), 0.0);
    
    // 在结尾添加哨兵值
    alphas_.push_back(static_cast<ERROR_TYPE>(sentinel_value));
    
    // printf("添加哨兵值后 alphas_.size() = %zu\n", alphas_.size());
  }
  
  return SUCCESS;
}



RESPONSE upcite::ConformalRegressor::fit_batch(const std::vector<std::vector<ERROR_TYPE>>& batch_residuals) {
  // 清空之前的批次alphas
  batch_alphas_.clear();
  // 将残差的绝对值存储到 alphas_ 中
  ID_TYPE num_batches = batch_residuals.size();
  batch_alphas_.resize(num_batches);

  // printf("[DEBUG] ConformalRegressor::fit_batch: 处理 %d 个批次\n", num_batches);
  // printf("batch_alphas_外层大小（批次数）: %zu\n", batch_alphas_.size());

  // 为每个批次分别排序并计算alphas
  for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    const auto& residuals = batch_residuals[batch_idx];
    if (residuals.size() < 3) {
      printf("[ERROR] 批次 %ld 样本不足: %zu (需要至少3个)\n", batch_idx+1, residuals.size());
      return FAILURE;
    }
    // 复制并排序此批次的残差
    std::vector<ERROR_TYPE> sorted_residuals = residuals;
    // 排序残差值
    std::sort(sorted_residuals.begin(), sorted_residuals.end());
    
    // 计算此批次的alphas
    std::vector<VALUE_TYPE> batch_alpha_values;
    batch_alpha_values.reserve(sorted_residuals.size());
    
    for (ID_TYPE i = 0; i < sorted_residuals.size(); ++i) {
      batch_alpha_values.push_back(sorted_residuals[i]);
    }
    batch_alphas_[batch_idx] = std::move(batch_alpha_values);
  }

  // 保存批处理alphas为CSV文件
  // std::string save_path = config.save_path + "/batch_alphas.csv";
  // save_batch_alphas_csv(save_path);
  return SUCCESS;
}



RESPONSE upcite::ConformalRegressor::fit_spline(std::string &spline_core, std::vector<ERROR_TYPE> &recalls) {
  std::cout << "spline_core = " << spline_core << std::endl;

  // 打印 alphas_ 的大小
  // printf("alphas_.size() = %ld\n", static_cast<long>(alphas_.size()));

  // 打印 alphas_ 的内容   
  // printf("alphas_ = [");
  // for (size_t i = 0; i < alphas_.size(); ++i) {
  //   printf("%.3f", static_cast<double>(alphas_[i]));
  //   if (i < alphas_.size() - 1) {
  //     printf(", ");
  //   }
  // }
  // printf("]\n");

  assert(recalls.size() == alphas_.size());
  gsl_accel_ = std::unique_ptr<gsl_interp_accel>(gsl_interp_accel_alloc());

  if (spline_core == "steffen") {  //选的是Steffen 样条
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  } else if (spline_core == "cubic") { //如果spline_core是"cubic"，则使用 三次样条（gsl_interp_cspline）
    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_cspline, recalls.size()));
  } else {
    spdlog::error("conformal spline core {:s} is not recognized; roll back to the default: steffen", spline_core);

    gsl_spline_ = std::unique_ptr<gsl_spline>(gsl_spline_alloc(gsl_interp_steffen, recalls.size()));
  }
  //初始化样条曲线
  gsl_spline_init(gsl_spline_.get(), recalls.data(), alphas_.data(), recalls.size());
  
  is_fitted_ = true; //表示样条曲线已成功拟合。
  is_trial_ = false; //表示当前不是试验模式

  return SUCCESS;
}





RESPONSE upcite::ConformalRegressor::fit_batch_spline(std::string &spline_core, std::vector<ERROR_TYPE> &avg_recalls) {
  // ================== 新增多批次处理逻辑 ================== //
  // 检查批次数据有效性
  if (batch_alphas_.empty()) {
    spdlog::error("Cannot fit spline without batch alpha data");
    return FAILURE;
  }

  // 验证所有批次的alpha数量一致
  const size_t num_quantiles = batch_alphas_[0].size();
  for (const auto& batch : batch_alphas_) {
    if (batch.size() != num_quantiles) {
      spdlog::error("Inconsistent quantile count across batches (expected {} got {})", 
                   num_quantiles, batch.size());
      return FAILURE;
    }
  }

  // 计算多批次平均alpha值
  std::vector<ERROR_TYPE> avg_alphas(num_quantiles, 0);
  for (size_t q = 0; q < num_quantiles; ++q) {
    double max_alpha = 0; // 初始化最大alpha值
    for (const auto& batch : batch_alphas_) { // 修正：使用batch_alphas_
      // 寻找最大的alpha值
      if (batch[q] > max_alpha) {
        max_alpha = batch[q];
      }
    }
    // 不再使用平均值，而是使用最大值
    avg_alphas[q] = max_alpha;
  }

  // 验证recall数据匹配
  if (avg_recalls.size() != num_quantiles) {
    spdlog::error("Recalls size mismatch (expected {} got {})", 
                 num_quantiles, avg_recalls.size());
    return FAILURE;
  }
  // ================== 修改结束 ================== //

  // 打印调试信息
  spdlog::debug("Fitting {} spline with {} quantiles", spline_core, num_quantiles);
  // std::cout << "spline_core = " << spline_core << std::endl;

  // 重建加速器和样条
  gsl_accel_.reset(gsl_interp_accel_alloc());
  
  // 选择样条类型
  const gsl_interp_type* spline_type = gsl_interp_steffen; // 默认
  if (spline_core == "cubic") {
    spline_type = gsl_interp_cspline;
  } else if (spline_core != "steffen") {
    spdlog::warn("Unsupported spline type: {}, using Steffen", spline_core);
  }

  // 创建样条对象
  gsl_spline_.reset(gsl_spline_alloc(spline_type, num_quantiles));

  // 初始化样条（使用平均后的数据）
  gsl_spline_init(gsl_spline_.get(), 
                 avg_recalls.data(), 
                 avg_alphas.data(),  // 使用计算的平均alpha
                 num_quantiles);

  // 打印拟合数据
  // #ifdef DEBUG
  // printf("[DEBUG] Spline fitting data:\n");
  // for (size_t i = 0; i < num_quantiles; ++i) {
  //   printf("Quantile %zu | Recall: %.4f | Alpha: %.4f\n", 
  //         i, avg_recalls[i], avg_alphas[i]);
  // }
  // #endif

  // 更新状态
  is_fitted_ = true;
  is_trial_ = false;

  return SUCCESS;
}


// 在conformal.cc中实现新方法
VALUE_TYPE upcite::ConformalRegressor::predict_calibrated(VALUE_TYPE y_hat) const {
  // 确保模型已训练
  if (!is_fitted_ && !is_trial_) {
    return y_hat; // 无法校准，直接返回原值
  }
  // 获取当前的alpha值
  VALUE_TYPE current_alpha = alpha_;
  // QYL: error修改为0
  // VALUE_TYPE current_alpha = 0;
  // 计算校准后的预测值（预测 - alpha）
  VALUE_TYPE calibrated = y_hat - current_alpha;
  // printf("predict = %.3f, alpha = %.3f, predict-alpha = %.3f\n", y_hat, current_alpha, calibrated);
  // 不允许负值
  calibrated = std::max(static_cast<VALUE_TYPE>(0.0), calibrated);
  // printf("predict_calibrated: 预测距离=%.3f, 预分配误差alpha=%.3f, 校准后距离=%.3f\n", y_hat, current_alpha, calibrated);
  // printf("CP: y_hat=%.3f, alpha=%.3f, y_hat-alpha=%.3f\n", y_hat, current_alpha, calibrated);
  // spdlog::info("CP sqrt: y_hat={:.3f}, alpha={:.3f}",  y_hat, current_alpha);
  // printf("alpha_ = %.3f\n", current_alpha);
  return calibrated;
}


upcite::INTERVAL upcite::ConformalRegressor::predict(VALUE_TYPE y_hat,
                                                     VALUE_TYPE confidence_level,
                                                     VALUE_TYPE y_max,
                                                     VALUE_TYPE y_min) {

  if (is_fitted_) {
    // printf("is_fitted_ = %d\n", is_fitted_);
    // printf("confidence_level_ = %.3f\n", confidence_level_);
    // printf("alphas_.size() = %zu\n", alphas_.size());
    // confidence_level_ > 1 denotes it is set externally
    if (confidence_level_ <= 1 && !upcite::is_equal(confidence_level_, confidence_level)) {
      assert(confidence_level >= 0 && confidence_level <= 1);
      
      abs_error_i_ = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(alphas_.size()) * confidence_level);
      alpha_ = alphas_[abs_error_i_];

      confidence_level_ = confidence_level;
    }
    // printf("y_hat = %.3f, alpha_ = %.3f\n", y_hat, alpha_);
    // spdlog::info("y_hat = {:.3f}, alpha_ = {:.3f}", y_hat, alpha_);
    return {y_hat - alpha_, y_hat + alpha_};

  } else if (is_trial_) {
    // printf("is_trial_ = %d\n", is_trial_);
    // printf("alpha_ = %.3f\n", alpha_);
    // spdlog::info("is_trial_ = %d, alpha_ = %.3f\n", is_trial_, alpha_);
    return {y_hat - alpha_, y_hat + alpha_};
  } else {
    return {y_min, y_max};
  }
}


RESPONSE upcite::ConformalPredictor::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&core_), sizeof(CONFORMAL_CORE));
  // alphas can be recalculated
//  ID_TYPE alphas_size = static_cast<ID_TYPE>(alphas_.size());
//  node_fos.write(reinterpret_cast<const char *>(&alphas_size), sizeof(ID_TYPE));
//  node_fos.write(reinterpret_cast<const char *>(alphas_.data()), sizeof(VALUE_TYPE) * alphas_.size());
  return SUCCESS;
}


RESPONSE upcite::ConformalPredictor::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_core_buf = reinterpret_cast<CONFORMAL_CORE *>(ifs_buf);
//  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
//  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  ID_TYPE read_nbytes = sizeof(CONFORMAL_CORE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  core_ = ifs_core_buf[0];

//  read_nbytes = sizeof(ID_TYPE);
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  ID_TYPE alphas_size = ifs_id_buf[0];
//  alphas_.reserve(alphas_size);
//
//  read_nbytes = sizeof(VALUE_TYPE) * alphas_size;
//  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
//  alphas_.insert(alphas_.begin(), ifs_value_buf, ifs_value_buf + alphas_size);

  return SUCCESS;
}

VALUE_TYPE upcite::ConformalPredictor::get_alpha() const {
  if (is_fitted_) {
    return alpha_;
  } else if (is_trial_) {
    return alpha_;
  } else {
    return constant::MAX_VALUE;
  }
}


RESPONSE upcite::ConformalPredictor::set_alpha(VALUE_TYPE alpha, bool is_trial, bool to_rectify) {
  if (is_trial) {
    if (is_fitted_) {
      spdlog::error("conformal model is already fitted; cannot run trial");
      return FAILURE;
    } else {
      alpha_ = alpha;

      is_trial_ = true;
    }
  } else if (is_fitted_) {
    if (to_rectify) {
      alpha_ = alpha;
    } else {
      spdlog::error("conformal model is already fitted; cannot directly adjust alpha");
      return FAILURE;
    }
  } else {
    alpha_ = alpha;
  }

  printf("进入set_alpha: is_fitted=%d\n", is_fitted_);
  // 函数结尾
  printf("设置结果: alpha=%.3f\n", alpha_);

  return SUCCESS;
}

VALUE_TYPE upcite::ConformalPredictor::get_batch_alpha_by_pos(ID_TYPE batch_i, ID_TYPE pos) const {
  if (batch_i >= batch_alphas_.size() || batch_alphas_[batch_i].empty()) {
    return constant::MAX_VALUE;
  }
  if (pos >= batch_alphas_[batch_i].size()) {
    return constant::MAX_VALUE;
  }
  return batch_alphas_[batch_i][pos];
}


//它的作用是根据给定的位置 pos，从 alphas_ 数组中获取对应的值（alpha
VALUE_TYPE upcite::ConformalPredictor::get_alpha_by_pos(ID_TYPE pos) const {
  // 检查 alphas_ 是否为空
  if (alphas_.empty()) {
    spdlog::error("alphas_ is empty in ConformalPredictor");
    printf("错误：alphas_为空，返回MAX_VALUE\n");
    return constant::MAX_VALUE;
  }
  // 检查 pos 是否合法
  if (pos < 0) {
    printf("错误：pos=%ld 为负数，返回MAX_VALUE\n", pos);
    return constant::MAX_VALUE;
  }
  if (static_cast<size_t>(pos) >= alphas_.size()) {
    printf("错误：pos=%ld 超出范围 alphas_.size()=%zu，返回MAX_VALUE\n", pos, alphas_.size());
    return constant::MAX_VALUE;
  }
  // 检查值是否有效
  VALUE_TYPE result = alphas_[pos];
  if (std::isnan(result) || std::isinf(result)) {
    printf("错误：alphas_[%ld]=%f 不是有效数值，返回MAX_VALUE\n", pos, result);
    return constant::MAX_VALUE;
  }
  
  // printf("alphas_[%ld]=%f\n", pos, result);
  return result;
}


RESPONSE upcite::ConformalPredictor::set_alpha_by_pos(ID_TYPE pos) {
  if (pos >= 0 && pos < alphas_.size()) {
    alpha_ = alphas_[pos];

    // TODO design a better workflow for is_fitted_
    is_fitted_ = true;
    confidence_level_ = EXT_DISCRETE; 
    return SUCCESS;
  }
  return FAILURE;
}



// 离散方法，给定误差位置，对所有batch的误差取平均找到对应位置的误差
RESPONSE upcite::ConformalPredictor::set_batch_alpha_by_pos(ID_TYPE pos) {
  // 检查批次数据是否为空
  if (batch_alphas_.empty()) {
    spdlog::error("No batch alpha data available");
    return FAILURE;
  }
  // 检查所有批次在pos位置的有效性
  for (const auto& batch : batch_alphas_) {
    if (pos < 0 || pos >= static_cast<ID_TYPE>(batch.size())) {
      spdlog::error("Invalid pos {:d} for batch alpha data", pos);
      return FAILURE;
    }
  }
  
  // 查找所有批次中pos位置的最大alpha值
  ERROR_TYPE max_alpha = 0;
  for (const auto& batch : batch_alphas_) {
    if (batch[pos] > max_alpha) {
      max_alpha = batch[pos];
    }
  }
  
  // 使用最大值而不是平均值
  alpha_ = max_alpha;

  // 更新状态标志
  is_fitted_ = true;
  confidence_level_ = EXT_DISCRETE;

  // 打印调试信息
  spdlog::debug("Set alpha at pos {:d} to {:.2f} (max value across {:d} batches)", 
                pos, alpha_, batch_alphas_.size());
  printf("设置位置 %ld 的 alpha 值为 %.2f (在 %ld 个批次中的最大值)\n", 
         pos, alpha_, batch_alphas_.size());
  return SUCCESS;
}  


RESPONSE upcite::ConformalRegressor::set_alpha_by_recall(VALUE_TYPE recall) {
  assert(gsl_accel_ != nullptr && gsl_spline_ != nullptr);
  alpha_ = gsl_spline_eval(gsl_spline_.get(), recall, gsl_accel_.get());
  printf("recall = %.3f, alpha_ = %.3f\n", static_cast<double>(recall), static_cast<double>(alpha_));
  // TODO design a better workflow for is_fitted_
  is_fitted_ = true;
  confidence_level_ = EXT_SPLINE;
  return SUCCESS;
}




//这个函数返回的是分位数索引（index），而不是实际的误差值
ID_TYPE upcite::ConformalRegressor::predict_quantile_from_bivariate(ERROR_TYPE target_recall, ERROR_TYPE target_coverage) const {
  if (!gsl_accel_ || !gsl_spline_) {
    spdlog::error("二元回归模型未初始化");
    return 0; // 返回默认值
  }
  // 使用权重组合特征
  double combined_feature = recall_weight_ * target_recall + coverage_weight_ * target_coverage;
  // 使用样条曲线预测分位数
  double predicted_quantile = gsl_spline_eval(gsl_spline_.get(), combined_feature, gsl_accel_.get());
  // 将预测值四舍五入到最近的整数
  ID_TYPE result = static_cast<ID_TYPE>(std::round(predicted_quantile));
  // 确保结果在有效范围内
  if (result < 0) {
    result = 0;
  }
  // 如果有batch_alphas_数据，则使用第一个批次的大小作为上限
  size_t max_quantile = 0;
  if (!batch_alphas_.empty() && !batch_alphas_[0].empty()) {
    max_quantile = batch_alphas_[0].size() - 1;
  } else if (!alphas_.empty()) {
    max_quantile = alphas_.size() - 1;
  }
  if (result > static_cast<ID_TYPE>(max_quantile)) {
    result = static_cast<ID_TYPE>(max_quantile);
  }
  spdlog::debug("预测分位数: 召回率={:.4f}, 覆盖率={:.4f} => 分位数={}", 
               target_recall, target_coverage, result);
  return result;
}




RESPONSE upcite::ConformalRegressor::train_regression_model_for_recall_coverage(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE filter_id) {
    
    printf("进入train_regression_model_for_recall_coverage, recalls：%zu, coverages：%zu, error_indices：%zu\n", recalls.size(), coverages.size(), error_indices.size());    
    if (recalls.size() != coverages.size() || recalls.size() != error_indices.size() || recalls.empty()) {
        printf("错误：输入数据数量不匹配或为空\n");
        return FAILURE;
    }
    // 获取当前filter下的cp类的batch_alphas，用于将位置转换为实际误差值
    const auto& batch_alphas = this->get_batch_alphas();
    if (batch_alphas.empty()) {
        printf("错误：batch_alphas为空，无法获取误差值\n");
        return FAILURE;
    }
    
    // printf("batch_alphas大小: %zu 批次, 每个批次 %zu 个误差值\n", batch_alphas.size(),  batch_alphas[0].size());
    spdlog::info("batch_alphas大小: {} 批次, 每个批次 {} 个误差值", batch_alphas.size(),  batch_alphas[0].size());
    
    // 准备训练数据，将error_i转换为对应的最大误差值
    std::vector<double> actual_errors;
    actual_errors.reserve(error_indices.size());

    for (ID_TYPE error_i : error_indices) {
        // printf("error_i %d\n", error_i);
        // 从所有批次中找出该位置对应的最大误差值
        double max_error = 0.0;
        //内层循环是为了找到error_i位置下的最大误差值（取同一个误差分位下的所有bacth的最大值）
        for (const auto& batch : batch_alphas) {
            // printf("当前batch大小: %zu\n", batch.size());  17
            if (error_i < batch.size() && batch[error_i] > max_error) {
                max_error = batch[error_i];
            }
        }
        // printf("max_error %.2f\n", max_error);
        // printf("\n");
        // actual_errors 是每个误差位置下的实际误差值（取所有bacth的最大值）
        actual_errors.push_back(max_error);
        // printf("actual_errors[%d]=%.6f\n", error_i, max_error);
    }
    
    // 打印actual_errors的大小
    // printf("过滤器 %ld 使用最大误差值的actual_errors大小: %zu\n", 
    //        (long)filter_id, actual_errors.size());
    spdlog::info("过滤器 {} 使用最大误差值的actual_errors大小: {}", 
                filter_id, actual_errors.size());

    // 打印所有actual_errors的取值
    // printf("actual_errors: ");
    // for (size_t i = 0; i < actual_errors.size(); i++) {
    //     printf("%.2f, ", actual_errors[i]);
    // }
    // printf("\n");

    // 样本数和特征数 - 根据实际样本量动态调整特征数
    size_t n = recalls.size();  // 样本数
    // 确定可用特征数量，确保样本数大于特征数
    // 定义可能的最大特征数（完整模型）
    const size_t max_features = 6;  // 常数项、recall、coverage、recall*coverage、recall^2、coverage^2
    // 实际使用的特征数量，需要确保 n > p
    size_t p = std::min(max_features, n > 2 ? n - 1 : 1);  // 确保至少保留1个特征（常数项）
    // printf("样本数: %zu, 使用特征数: %zu\n", n, p);
    spdlog::info("样本数: {}, 使用特征数: {}", n, p);
    
    // 使用GSL库进行多元回归
    gsl_matrix *X = gsl_matrix_alloc(n, p);
    gsl_vector *y = gsl_vector_alloc(n);
    gsl_vector *c = gsl_vector_alloc(p);  // 回归系数
    gsl_matrix *cov = gsl_matrix_alloc(p, p);  // 协方差矩阵
    double chisq;  // 拟合优度
    
    // 填充设计矩阵X和目标向量y（注意：y现在是实际误差值）
    for (size_t i = 0; i < n; i++) {
        // 始终添加常数项
        gsl_matrix_set(X, i, 0, 1.0);  // 常数项
        // 根据可用特征数量逐步添加更复杂的特征
        if (p > 1) gsl_matrix_set(X, i, 1, recalls[i]);
        if (p > 2) gsl_matrix_set(X, i, 2, coverages[i]);
        if (p > 3) gsl_matrix_set(X, i, 3, recalls[i] * coverages[i]);  // 交互项
        if (p > 4) gsl_matrix_set(X, i, 4, recalls[i] * recalls[i]);
        if (p > 5) gsl_matrix_set(X, i, 5, coverages[i] * coverages[i]);
        
        // 目标向量 - 使用实际误差值
        gsl_vector_set(y, i, actual_errors[i]);
    }
    
    // 执行回归计算
    gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, p);
    int ret = gsl_multifit_linear(X, y, c, cov, &chisq, work);
    
    if (ret != GSL_SUCCESS) {
        printf("错误：GSL回归计算失败，错误码：%d\n", ret);
        // 清理资源
        gsl_multifit_linear_free(work);
        gsl_matrix_free(X);
        gsl_vector_free(y);
        gsl_vector_free(c);
        gsl_matrix_free(cov);
        return FAILURE;
    }
    
    // 提取回归系数 - 确保存储完整的6个系数，未使用的设为0
    regression_coeffs_.resize(max_features, 0.0); // 初始化为全0
    for (size_t i = 0; i < p; i++) {
        regression_coeffs_[i] = gsl_vector_get(c, i);
    }
    
    // 打印回归系数和拟合优度
    printf("回归系数: [常数项, recall, coverage, recall*coverage, recall^2, coverage^2]\n");
    printf("         [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
           regression_coeffs_[0], regression_coeffs_[1], regression_coeffs_[2],
           regression_coeffs_[3], regression_coeffs_[4], regression_coeffs_[5]);

    printf("拟合优度 (chi^2): %.1f\n", chisq);
    spdlog::info("拟合优度 (chi^2): {}", chisq);
    // 清理GSL资源
    gsl_multifit_linear_free(work);
    gsl_matrix_free(X);
    gsl_vector_free(y);
    gsl_vector_free(c);
    gsl_matrix_free(cov);
    
    // 更新模型状态
    is_fitted_ = true;
    return SUCCESS;
}



// 添加清理批处理数据的方法
void upcite::ConformalPredictor::clear_batch_data() {
    // 只清空数据，不强制释放内存
    for (auto& batch : batch_alphas_) {
        batch.clear();
    }
    batch_alphas_.clear();
}


// 保存函数修改
RESPONSE upcite::ConformalPredictor::save_batch_alphas(const std::string& filepath) const {
    // 检查是否有批次数据
    if (batch_alphas_.empty()) {
        spdlog::error("No batch alpha data to save");
        printf("没有批次alpha数据可保存\n");
        return FAILURE;
    }
    // 使用文本模式打开文件
    std::ofstream alphas_fout(filepath);  // 移除 std::ios::binary
    if (!alphas_fout.good()) {
        spdlog::error("Failed to open file for writing: {}", filepath);
        printf("无法打开文件进行写入: %s\n", filepath.c_str());
        return FAILURE;
    }
    // 写入类型信息和批次数量
    alphas_fout << "TYPE_SIZE=" << sizeof(ERROR_TYPE) << std::endl;
    alphas_fout << "NUM_BATCHES=" << batch_alphas_.size() << std::endl;
    // 对每个批次
    for (size_t batch_i = 0; batch_i < batch_alphas_.size(); ++batch_i) {
        // 写入批次大小
        alphas_fout << "BATCH_" << batch_i << "_SIZE=" << batch_alphas_[batch_i].size() << std::endl;
        // 逐个写入值，使用科学计数法确保精度
        for (size_t j = 0; j < batch_alphas_[batch_i].size(); ++j) {
            ERROR_TYPE val = batch_alphas_[batch_i][j];
            // 检查值的有效性
            if (std::isnan(val) || std::isinf(val) || val < 0 || val > 1e10) {
                printf("警告：保存批次%zu索引%zu的值%.6g无效\n", batch_i, j, (double)val);
            }
            // 使用科学计数法，保证15位有效数字
            alphas_fout << std::scientific << std::setprecision(15) << val;
            // 每行结束添加换行符
            alphas_fout << std::endl;
        }
    }
    // printf("成功保存批次alpha值到: %s (文本格式)\n", filepath.c_str());
    return SUCCESS;
}

// 保存批处理alphas为CSV文件（行=批次，列=alpha序号）
RESPONSE upcite::ConformalPredictor::save_batch_alphas_csv(const std::string& filepath) const {
    if (batch_alphas_.empty()) {
        spdlog::error("[save_batch_alphas_csv] No batch alpha data to save");
        return FAILURE;
    }
    std::ofstream fout(filepath);
    if (!fout.is_open()) {
        spdlog::error("[save_batch_alphas_csv] Cannot open file {}", filepath);
        return FAILURE;
    }
    for (size_t batch_i = 0; batch_i < batch_alphas_.size(); ++batch_i) {
        const auto &vec = batch_alphas_[batch_i];
        for (size_t j = 0; j < vec.size(); ++j) {
            fout << vec[j];
            if (j + 1 < vec.size()) fout << ",";
        }
        fout << "\n";
    }
    fout.close();
    spdlog::info("[save_batch_alphas_csv] saved CSV to {} (batches={})", filepath, batch_alphas_.size());
    // printf("[save_batch_alphas_csv] saved CSV to %s (batches=%zu)\n", filepath.c_str(), batch_alphas_.size());
    return SUCCESS;
}


// 加载函数修改
RESPONSE upcite::ConformalPredictor::load_batch_alphas(const std::string& filepath) {
    // 完全重置数据，确保没有旧数据
    batch_alphas_.clear();
    // 使用文本模式打开文件
    std::ifstream file(filepath);
    if (!file) {
        printf("无法打开文件: %s\n", filepath.c_str());
        return FAILURE;
    }
    // printf("加载alpha文件: %s (文本格式)\n", filepath.c_str());
    // 读取类型大小和批次数量
    std::string line;
    size_t saved_type_size = 0;
    size_t num_batches = 0;
    // 读取类型大小行
    if (!std::getline(file, line) || sscanf(line.c_str(), "TYPE_SIZE=%zu", &saved_type_size) != 1) {
        printf("文件格式错误: 无法读取类型大小\n");
        return FAILURE;
    }
    // 读取批次数量行
    if (!std::getline(file, line) || sscanf(line.c_str(), "NUM_BATCHES=%zu", &num_batches) != 1) {
        printf("文件格式错误: 无法读取批次数量\n");
        return FAILURE;
    }
    // printf("文件中保存的ERROR_TYPE大小: %zu 字节, 批次数量: %zu\n", 
    //        saved_type_size, num_batches);
    // 检查批次数量的合理性
    if (num_batches <= 0 || num_batches > 1000) {
        printf("无效的批次数量: %zu\n", num_batches);
        return FAILURE;
    }
    // 预分配空间
    try {
        batch_alphas_.resize(num_batches);
    } catch (const std::exception& e) {
        printf("分配内存失败: %s\n", e.what());
        return FAILURE;
    }
    
    // 解析每个批次的数据
    for (size_t batch_i = 0; batch_i < num_batches; ++batch_i) {
        // 读取批次大小行
        size_t batch_size = 0;
        if (!std::getline(file, line) || 
            sscanf(line.c_str(), "BATCH_%zu_SIZE=%zu", &batch_i, &batch_size) != 2) {
            printf("文件格式错误: 无法读取批次%zu大小\n", batch_i);
            return FAILURE;
        }
        
        // 检查批次大小的合理性
        if (batch_size <= 0 || batch_size > 10000) {
            printf("无效的批次大小: %zu (批次 %zu)\n", batch_size, batch_i);
            return FAILURE;
        }
        
        // printf("加载批次 %zu, 大小为 %zu\n", batch_i, batch_size);
        
        // 预分配批次数据空间
        batch_alphas_[batch_i].resize(batch_size);
        
        // 逐行读取值
        int invalid_count = 0;
        for (size_t j = 0; j < batch_size; ++j) {
            if (!std::getline(file, line)) {
                printf("文件格式错误: 批次%zu数据不完整\n", batch_i);
                return FAILURE;
            }
            // 将文本转换为double
            double val = std::stod(line);
            // 检查数据有效性
            if (std::isnan(val) || std::isinf(val) || val < 0 || val > 1e10) {
                // 无效值用0替代
                if (invalid_count < 10) {
                    printf("批次%zu索引%ld的值%.6g无效, 替换为0\n", batch_i, (long)j, val);
                } else if (invalid_count == 10) {
                    printf("更多无效值...\n");
                }
                batch_alphas_[batch_i][j] = 0.0;
                invalid_count++;
            } else {
                // 有效值正常保存
                batch_alphas_[batch_i][j] = static_cast<ERROR_TYPE>(val);
            }
        }
        if (invalid_count > 0) {
            printf("警告：批次 %zu 包含 %d 个无效数据\n", batch_i, invalid_count);
        }
    }
    // 完成解析，标记为已拟合
    is_fitted_ = true;
    // printf("成功从 %s 加载批次alpha值\n", filepath.c_str());
    return SUCCESS;
}




RESPONSE upcite::ConformalRegressor::set_alpha_by_recall_and_coverage(ERROR_TYPE target_recall, ERROR_TYPE target_coverage) {
  if (!is_fitted_) {
    spdlog::error("Cannot set alpha, model not fitted yet");
    printf("设置alpha失败: 模型未训练\n");
    return FAILURE;
  }
  alpha_ = predict_error_value(target_recall, target_coverage);
  // alpha_ = predict_eigen_quadratic_spline(target_recall, target_coverage, regression_coeffs_);
  spdlog::info("设置 alpha 为 {:.3f}", alpha_);
  // printf("设置 alpha=%.3f\n",  alpha_);
  // 更新状态
  is_fitted_ = true;
  confidence_level_ = EXT_SPLINE; // 使用特殊标记表示外部设置的置信度
  
  return SUCCESS;
}



//eigenspline回归
double upcite::ConformalRegressor::predict_error_value(double recall, double coverage) const {
    if (!is_fitted_) {
        spdlog::warn("回归模型未训练");
        printf("回归模型未训练\n");
        return -1.0;
    }
    // 直接使用Eigen样条模型进行预测
    double predicted_error = predict_eigen_quadratic_spline(recall, coverage, regression_coeffs_);
    // double predicted_error = predict_alglib_quadratic_spline(recall, coverage, regression_coeffs_);
     // spdlog::info("Eigen spline 的 alpha: {:.4f}, recall={:.4f}, coverage={:.4f}", 
    //             std::max(0.0, predicted_error), recall, coverage); 
    // printf("alpha: %.4f, recall=%.2f, coverage=%.2f\n", 
    //             std::max(0.0, predicted_error), recall, coverage); 
    return std::max(0.0, predicted_error);
}

// 提供对外部调用友好的接口
// ERROR_TYPE upcite::ConformalRegressor::predict_eigen_spline_error(
//     ERROR_TYPE recall, ERROR_TYPE coverage) const {
//     // 使用已存储的回归系数进行预测
//     return predict_eigen_quadratic_spline(recall, coverage, regression_coeffs_);
// }


RESPONSE upcite::ConformalRegressor::fit_eigen_quadratic_spline(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages, 
    const std::vector<ERROR_TYPE>& errors,
    std::vector<double>& model_coeffs) {
    
    spdlog::info("开始构建网格插值模型...");
    
    if (recalls.size() != coverages.size() || recalls.size() != errors.size() || recalls.empty()) {
        spdlog::error("输入数据维度不匹配或为空");
        return FAILURE;
    }
    
    // 先复制原始数据并确保单调性
    std::vector<ERROR_TYPE> filtered_recalls = recalls;
    std::vector<ERROR_TYPE> filtered_coverages = coverages;
    std::vector<ERROR_TYPE> filtered_errors = errors;
    filter_monotonic_data(filtered_recalls, filtered_coverages, filtered_errors);
    
    // 准备数据点
    const size_t n_points = filtered_recalls.size();
    
    try {
        // 计算点的边界
        double min_recall = *std::min_element(filtered_recalls.begin(), filtered_recalls.end());
        double max_recall = *std::max_element(filtered_recalls.begin(), filtered_recalls.end());
        double min_coverage = *std::min_element(filtered_coverages.begin(), filtered_coverages.end());
        double max_coverage = *std::max_element(filtered_coverages.begin(), filtered_coverages.end());
        
        // 创建参数化的控制点网格 (更密集的网格以获得更好的平滑度)
        const int grid_size = std::min(20, static_cast<int>(std::sqrt(n_points)) + 5);
        Eigen::MatrixXd grid(grid_size * grid_size, 3);
        
        // 创建均匀网格
        int idx = 0;
        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                double u = min_recall + (max_recall - min_recall) * i / (grid_size - 1);
                double v = min_coverage + (max_coverage - min_coverage) * j / (grid_size - 1);
                
                // === 优化部分：为网格点找最近的K个原始数据点，取最大error ===
                const int K = 5; // 使用最近的3个点
                std::vector<std::pair<double, size_t>> nearest_points;
                nearest_points.reserve(n_points);
                
                // 计算所有点的距离
                for (size_t k = 0; k < n_points; k++) {
                    double d = std::pow(u - filtered_recalls[k], 2) + 
                               std::pow(v - filtered_coverages[k], 2);
                    nearest_points.push_back({d, k});
                }
                
                // 只保留最近的K个点
                std::partial_sort(nearest_points.begin(), 
                                 nearest_points.begin() + std::min(K, (int)nearest_points.size()), 
                                 nearest_points.end());
                
                // 找出最近K个点中error最大的
                double max_error = -std::numeric_limits<double>::max();
                for (int k = 0; k < std::min(K, (int)nearest_points.size()); k++) {
                    size_t idx_k = nearest_points[k].second;
                    max_error = std::max(max_error, filtered_errors[idx_k]);
                }
                
                // 添加到网格中，使用找到的最大error值
                grid(idx, 0) = u;
                grid(idx, 1) = v;
                grid(idx, 2) = max_error;
                idx++;
            }
        }
        
        // === 精简部分：删除不必要的Eigen::SplineFitting代码 ===
        
        // 存储网格数据到model_coeffs
        model_coeffs.clear();
        
        // 存储基本参数
        model_coeffs.push_back(min_recall);
        model_coeffs.push_back(max_recall);
        model_coeffs.push_back(min_coverage);
        model_coeffs.push_back(max_coverage);
        model_coeffs.push_back(2); // 保留spline_degree以维持与现有predict函数兼容
        model_coeffs.push_back(grid_size);
        
        // 存储网格点
        for (int i = 0; i < grid.rows(); i++) {
            model_coeffs.push_back(grid(i, 0)); // recall
            model_coeffs.push_back(grid(i, 1)); // coverage
            model_coeffs.push_back(grid(i, 2)); // error
        }
        
        // 更新回归系数并标记模型已拟合
        regression_coeffs_ = model_coeffs;
        is_fitted_ = true;
        
        spdlog::info("网格插值模型创建成功！存储了{}个参数", model_coeffs.size());
        return SUCCESS;
    }
    catch (const std::exception& e) {
        spdlog::error("网格插值模型创建失败: {}", e.what());
        return FAILURE;
    }
}



// 修复fit_eigen_quadratic_spline函数
// RESPONSE upcite::ConformalRegressor::fit_eigen_quadratic_spline(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages, 
//     const std::vector<ERROR_TYPE>& errors,
//     std::vector<double>& model_coeffs) {
    
//     spdlog::info("开始使用Eigen Splines模块拟合二维样条模型...");
    
//     if (recalls.size() != coverages.size() || recalls.size() != errors.size() || recalls.empty()) {
//         spdlog::error("输入数据维度不匹配或为空");
//         return FAILURE;
//     }
    
//     // 先复制原始数据并确保单调性
//     std::vector<ERROR_TYPE> filtered_recalls = recalls;
//     std::vector<ERROR_TYPE> filtered_coverages = coverages;
//     std::vector<ERROR_TYPE> filtered_errors = errors;
//     filter_monotonic_data(filtered_recalls, filtered_coverages, filtered_errors);
    
//     // 准备数据点
//     const size_t n_points = filtered_recalls.size();
    
//     try {
//         // 计算点的边界
//         double min_recall = *std::min_element(filtered_recalls.begin(), filtered_recalls.end());
//         double max_recall = *std::max_element(filtered_recalls.begin(), filtered_recalls.end());
//         double min_coverage = *std::min_element(filtered_coverages.begin(), filtered_coverages.end());
//         double max_coverage = *std::max_element(filtered_coverages.begin(), filtered_coverages.end());
        
//         // 创建参数化的控制点网格 (更密集的网格以获得更好的平滑度)
//         const int grid_size = std::min(20, static_cast<int>(std::sqrt(n_points)) + 5);
//         Eigen::MatrixXd grid(grid_size * grid_size, 3);
        
//         // 创建均匀网格
//         int idx = 0;
//         for (int i = 0; i < grid_size; i++) {
//             for (int j = 0; j < grid_size; j++) {
//                 double u = min_recall + (max_recall - min_recall) * i / (grid_size - 1);
//                 double v = min_coverage + (max_coverage - min_coverage) * j / (grid_size - 1);
                
//                 // 为网格点找最近的原始数据点
//                 double min_dist = std::numeric_limits<double>::max();
//                 size_t nearest_idx = 0;
                
//                 for (size_t k = 0; k < n_points; k++) {
//                     double d = std::pow(u - filtered_recalls[k], 2) + 
//                                std::pow(v - filtered_coverages[k], 2);
//                     if (d < min_dist) {
//                         min_dist = d;
//                         nearest_idx = k;
//                     }
//                 }
                
//                 // 添加到网格中，使用最近点的误差值
//                 grid(idx, 0) = u;
//                 grid(idx, 1) = v;
//                 grid(idx, 2) = filtered_errors[nearest_idx];
//                 idx++;
//             }
//         }
        
//         // 修正：按照Eigen API的格式构建点集合
//         // 正确的格式：(dims+1) x n_points 矩阵，前dims行是坐标，最后一行是值
//         Eigen::MatrixXd points_data(3, grid_size * grid_size);
//         for (int i = 0; i < grid_size * grid_size; i++) {
//             points_data(0, i) = grid(i, 0);  // recall
//             points_data(1, i) = grid(i, 1);  // coverage
//             points_data(2, i) = grid(i, 2);  // error
//         }
        
//         // 使用Eigen::SplineFitting来拟合样条 - 只使用两个参数
//         const int spline_degree = 2; // 二次样条
//         typedef Eigen::Spline<double, 2> Spline2d;
//         auto spline = Eigen::SplineFitting<Spline2d>::Interpolate(points_data, spline_degree);
            
//         // 存储样条系数到model_coeffs
//         // Eigen没有直接提供访问系数的方法，我们需要存储控制点和度数信息
//         model_coeffs.clear();
        
//         // 存储基本参数
//         model_coeffs.push_back(min_recall);
//         model_coeffs.push_back(max_recall);
//         model_coeffs.push_back(min_coverage);
//         model_coeffs.push_back(max_coverage);
//         model_coeffs.push_back(spline_degree);
//         model_coeffs.push_back(grid_size);
        
//         // 存储样条的控制点
//         for (int i = 0; i < grid.rows(); i++) {
//             model_coeffs.push_back(grid(i, 0)); // recall
//             model_coeffs.push_back(grid(i, 1)); // coverage
//             model_coeffs.push_back(grid(i, 2)); // error
//         }
        
//         // 更新回归系数并标记模型已拟合
//         regression_coeffs_ = model_coeffs;
//         is_fitted_ = true;
        
//         spdlog::info("Eigen样条模型创建成功！存储了{}个参数", model_coeffs.size());
//         return SUCCESS;
//     }
//     catch (const std::exception& e) {
//         spdlog::error("Eigen样条拟合失败: {}", e.what());
//         return FAILURE;
//     }
// }


// 修复predict_eigen_quadratic_spline函数
// ERROR_TYPE upcite::ConformalRegressor::predict_eigen_quadratic_spline(
//     ERROR_TYPE recall, 
//     ERROR_TYPE coverage, 
//     const std::vector<double>& model_coeffs) const {
    
//     if (!is_fitted_ || model_coeffs.empty()) {
//         spdlog::error("模型未拟合或模型系数为空");
//         return 0.0;
//     }
//     // 至少需要6个基本参数 + 网格点数据
//     if (model_coeffs.size() < 6) {
//         spdlog::error("模型系数不完整，无法预测");
//         return 0.0;
//     }
//     // 1. 提取基本参数
//     double min_recall = model_coeffs[0];
//     double max_recall = model_coeffs[1];
//     double min_coverage = model_coeffs[2];
//     double max_coverage = model_coeffs[3];
//     int spline_degree = static_cast<int>(model_coeffs[4]);
//     int grid_size = static_cast<int>(model_coeffs[5]);
    
//     // 2. 处理超出范围的输入值
//     ERROR_TYPE bounded_recall = std::max(min_recall, std::min(max_recall, recall));
//     ERROR_TYPE bounded_coverage = std::max(min_coverage, std::min(max_coverage, coverage));
    
//     // 3. 重建网格数据
//     const int expected_coeffs_size = 6 + grid_size * grid_size * 3;
//     if (model_coeffs.size() < expected_coeffs_size) {
//         spdlog::error("模型系数大小({})小于期望大小({})", model_coeffs.size(), expected_coeffs_size);
//         return 0.0;
//     }
    
//     Eigen::MatrixXd grid(grid_size * grid_size, 3);
//     for (int i = 0; i < grid_size * grid_size; i++) {
//         grid(i, 0) = model_coeffs[6 + i * 3];     // recall
//         grid(i, 1) = model_coeffs[6 + i * 3 + 1]; // coverage
//         grid(i, 2) = model_coeffs[6 + i * 3 + 2]; // error
//     }
    
//     try {
//         // 4. 构建点集合 - 正确的格式：(dims+1) x n_points
//         Eigen::MatrixXd points_data(3, grid_size * grid_size);
//         for (int i = 0; i < grid_size * grid_size; i++) {
//             points_data(0, i) = grid(i, 0);  // recall
//             points_data(1, i) = grid(i, 1);  // coverage
//             points_data(2, i) = grid(i, 2);  // error
//         }
        
//         // 5. 使用Eigen::SplineFitting创建样条
//         typedef Eigen::Spline<double, 2> Spline2d;
//         auto spline = Eigen::SplineFitting<Spline2d>::Interpolate(points_data, spline_degree);
        
//         // 6. 要注意：Eigen的Spline只接受单一double参数u∈[0,1]
//         // 我们需要将2D点映射到1D参数
//         double u = (bounded_recall - min_recall) / (max_recall - min_recall);
//         u = std::max(0.0, std::min(1.0, u)); // 确保在[0,1]范围内
        
//         // 7. 基于u内插找到对应的样条值
//         // 注意：这是一个简化方法，因为Eigen的Spline API限制
//         double predicted_error = 0.0;
//         double total_weight = 0.0;
        
//         // 使用Shepard插值（基于距离的加权平均）
//         for (int i = 0; i < grid_size * grid_size; i++) {
//             double r = grid(i, 0);
//             double c = grid(i, 1);
//             double e = grid(i, 2);
            
//             double dist = std::pow(r - bounded_recall, 2) + std::pow(c - bounded_coverage, 2);
//             if (dist < 1e-10) {
//                 // 如果距离非常小，直接返回该点的误差值
//                 return std::max(0.001, 0.2 + e * 1.5);
//             }
            
//             double weight = 1.0 / dist;
//             total_weight += weight;
//             predicted_error += weight * e;
//         }
        
//         if (total_weight > 0) {
//             predicted_error /= total_weight;
//         }
        
//         // 8. 添加基础值和校正因子
//         double base_alpha = 0.2;
//         double correction_factor = 1.5;
//         double final_prediction = base_alpha + predicted_error * correction_factor;
        
//         // 9. 确保预测值为正
//         return std::max(0.001, final_prediction);
//     }
//     catch (const std::exception& e) {
//         spdlog::error("样条预测失败: {}", e.what());
        
//         // 10. 失败时回退到简单的最近邻
//         double min_dist = std::numeric_limits<double>::max();
//         double nearest_error = 0.0;
        
//         for (int i = 0; i < grid_size * grid_size; i++) {
//             double r = grid(i, 0);
//             double c = grid(i, 1);
//             double e = grid(i, 2);
            
//             double dist = std::pow(r - bounded_recall, 2) + std::pow(c - bounded_coverage, 2);
//             if (dist < min_dist) {
//                 min_dist = dist;
//                 nearest_error = e;
//             }
//         }
//         // 添加基础值和校正
//         return std::max(0.05, 0.4 + nearest_error * 2);
//     }
// }


// //这个版本是考虑保证预测error具有单调性
ERROR_TYPE upcite::ConformalRegressor::predict_eigen_quadratic_spline(
    ERROR_TYPE recall, 
    ERROR_TYPE coverage, 
    const std::vector<double>& model_coeffs) const {
    
    if (!is_fitted_ || model_coeffs.empty()) {
        spdlog::error("模型未拟合或模型系数为空");
        return 0.0;
    }
    
    // 至少需要6个基本参数 + 网格点数据
    if (model_coeffs.size() < 6) {
        spdlog::error("模型系数不完整，无法预测");
        return 0.0;
    }
    
    // 1. 提取基本参数
    double min_recall = model_coeffs[0];
    double max_recall = model_coeffs[1];
    double min_coverage = model_coeffs[2];
    double max_coverage = model_coeffs[3];
    int spline_degree = static_cast<int>(model_coeffs[4]); // 保留但不使用
    int grid_size = static_cast<int>(model_coeffs[5]);
    
    // 2. 处理超出范围的输入值
    ERROR_TYPE bounded_recall = std::max(min_recall, std::min(max_recall, recall));
    ERROR_TYPE bounded_coverage = std::max(min_coverage, std::min(max_coverage, coverage));
    
    // 3. 重建网格数据
    const int expected_coeffs_size = 6 + grid_size * grid_size * 3;
    if (model_coeffs.size() < expected_coeffs_size) {
        spdlog::error("模型系数大小({})小于期望大小({})", model_coeffs.size(), expected_coeffs_size);
        return 0.0;
    }
    
    try {
        // 重建网格点
        std::vector<std::tuple<double, double, double>> grid_points;
        grid_points.reserve(grid_size * grid_size);
        
        for (int i = 0; i < grid_size * grid_size; i++) {
            double r = model_coeffs[6 + i * 3];
            double c = model_coeffs[6 + i * 3 + 1];
            double e = model_coeffs[6 + i * 3 + 2];
            grid_points.emplace_back(r, c, e);
        }
        
        // 1. 强制单调性的辅助数据结构
        std::map<double, std::map<double, double>> monotonic_grid;
        
        // 首先填充原始数据
        for (const auto& [r, c, e] : grid_points) {
            monotonic_grid[r][c] = e;
        }
        
        // 2. 强制单调性 - 确保每个r行内随c增加而递增
        for (auto& [r, row] : monotonic_grid) {
            double max_so_far = -std::numeric_limits<double>::max();
            std::vector<double> c_values;
            for (const auto& [c, _] : row) c_values.push_back(c);
            std::sort(c_values.begin(), c_values.end());
            
            for (const auto& c : c_values) {
                max_so_far = std::max(max_so_far, row[c]);
                row[c] = max_so_far;
            }
        }
        
        // 3. 强制单调性 - 确保每个c列内随r增加而递增
        std::set<double> all_cs;
        for (const auto& [_, row] : monotonic_grid) {
            for (const auto& [c, __] : row) all_cs.insert(c);
        }
        
        std::vector<double> r_values;
        for (const auto& [r, _] : monotonic_grid) r_values.push_back(r);
        std::sort(r_values.begin(), r_values.end());
        
        for (const auto& c : all_cs) {
            double max_so_far = -std::numeric_limits<double>::max();
            for (const auto& r : r_values) {
                if (monotonic_grid[r].count(c)) {
                    max_so_far = std::max(max_so_far, monotonic_grid[r][c]);
                    monotonic_grid[r][c] = max_so_far;
                }
            }
        }
        
        // 4. 使用修正后的单调网格点
        std::vector<std::tuple<double, double, double>> monotonic_points;
        for (const auto& [r, row] : monotonic_grid) {
            for (const auto& [c, e] : row) {
                monotonic_points.emplace_back(r, c, e);
            }
        }
        
        // 5. 特殊处理高recall/coverage区域
        double base_boost = 0.0;
        if (bounded_recall >= 0.97) base_boost += (bounded_recall - 0.97) * 10.0;
        if (bounded_coverage >= 0.97) base_boost += (bounded_coverage - 0.97) * 5.0;
        
        // 6. 改进的Shepard插值
        double predicted_error = 0.0;
        double total_weight = 0.0;
        
        // 使用Shepard插值 - 基于单调网格
        for (const auto& [r, c, e] : monotonic_points) {
            double dist = std::pow(r - bounded_recall, 2) + std::pow(c - bounded_coverage, 2);
            if (dist < 1e-10) {
                return std::max(0.1, e + base_boost); // 增加0%作为安全边际
            }
            
            // 使用距离的立方倒数作为权重，进一步强调近点影响
            double weight = 1.0 / (dist * dist);//* dist
            total_weight += weight;
            predicted_error += weight * e;
        }
        
        if (total_weight > 0) {
            predicted_error /= total_weight;
        }
        
        // 7. 添加基础值和校正因子
        double base_alpha = 0.2; // 增加基础值
        double correction_factor = 1.4; // 增加校正因子
        double final_prediction = base_alpha + predicted_error * correction_factor + base_boost;
        
        // 8. 确保最小值
        return std::max(0.1, final_prediction);
    }
    catch (const std::exception& e) {
        spdlog::error("预测失败: {}", e.what());
        return 0.8; // 返回更保守的默认值
    }
}







// // // 这个版本是ok的，单纯利用网格点插值的，但是没有搞定单调性
// ERROR_TYPE upcite::ConformalRegressor::predict_eigen_quadratic_spline(
//     ERROR_TYPE recall, 
//     ERROR_TYPE coverage, 
//     const std::vector<double>& model_coeffs) const {
    
//     if (!is_fitted_ || model_coeffs.empty()) {
//         spdlog::error("模型未拟合或模型系数为空");
//         return 0.0;
//     }
    
//     // 1. 提取基本参数
//     double min_recall = model_coeffs[0];
//     double max_recall = model_coeffs[1];
//     double min_coverage = model_coeffs[2];
//     double max_coverage = model_coeffs[3];
//     int spline_degree = static_cast<int>(model_coeffs[4]); // 保留但不使用
//     int grid_size = static_cast<int>(model_coeffs[5]);
    
//     // 2. 处理超出范围的输入值
//     ERROR_TYPE bounded_recall = std::max(min_recall, std::min(max_recall, recall));
//     ERROR_TYPE bounded_coverage = std::max(min_coverage, std::min(max_coverage, coverage));
    
//     // 3. 重建网格数据
//     const int expected_coeffs_size = 6 + grid_size * grid_size * 3;
//     if (model_coeffs.size() < expected_coeffs_size) {
//         spdlog::error("模型系数大小({})小于期望大小({})", model_coeffs.size(), expected_coeffs_size);
//         return 0.0;
//     }
    
//     try {
//         // 4. 重建网格
//         std::vector<std::tuple<double, double, double>> grid_points;
//         grid_points.reserve(grid_size * grid_size);
        
//         for (int i = 0; i < grid_size * grid_size; i++) {
//             double r = model_coeffs[6 + i * 3];     // recall
//             double c = model_coeffs[6 + i * 3 + 1]; // coverage
//             double e = model_coeffs[6 + i * 3 + 2]; // error
//             grid_points.emplace_back(r, c, e);
//         }
        
//         // 5. 直接使用Shepard插值
//         double predicted_error = 0.0;
//         double total_weight = 0.0;
        
//         // 单调性强制变量
//         double min_r_greater_error = std::numeric_limits<double>::max();
//         double min_c_greater_error = std::numeric_limits<double>::max();
        
//         for (const auto& [r, c, e] : grid_points) {
//             // 找出recall相同但coverage更小的点中最大的error
//             if (std::abs(r - bounded_recall) < 1e-6 && c < bounded_coverage) {
//                 min_r_greater_error = std::min(min_r_greater_error, e);
//             }
            
//             // 找出coverage相同但recall更小的点中最大的error
//             if (std::abs(c - bounded_coverage) < 1e-6 && r < bounded_recall) {
//                 min_c_greater_error = std::min(min_c_greater_error, e);
//             }
            
//             double dist = std::pow(r - bounded_recall, 2) + std::pow(c - bounded_coverage, 2);
//             if (dist < 1e-10) {
//                 // 如果距离非常小，直接返回该点的误差值
//                 return std::max(0.001, 0.2 + e * 1.5);
//             }
            
//             // 使用距离的平方倒数作为权重，对远点的惩罚更大
//             double weight = 1.0 / (dist * dist); 
//             total_weight += weight;
//             predicted_error += weight * e;
//         }
        
//         if (total_weight > 0) {
//             predicted_error /= total_weight;
//         }
        
//         // 6. 单调性检查（可选，如果插值结果不单调可以启用）
//         if (min_r_greater_error != std::numeric_limits<double>::max() && 
//             predicted_error < min_r_greater_error) {
//             predicted_error = min_r_greater_error * 1.05; // 稍微增加以保持单调性
//         }
        
//         // 7. 添加基础值和校正因子
//         double base_alpha = 0.2;
//         double correction_factor = 1.4;
//         double final_prediction = base_alpha + predicted_error * correction_factor;
        
//         // 8. 确保预测值为正
//         return std::max(0.001, final_prediction);
//     }
//     catch (const std::exception& e) {
//         spdlog::error("预测失败: {}", e.what());
//         return 0.5; // 返回一个合理的默认值
//     }
// }


// 添加头文件
// ALGLIB二维样条拟合函数 - 使用RBF模型
RESPONSE upcite::ConformalRegressor::fit_alglib_quadratic_spline(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages, 
    const std::vector<ERROR_TYPE>& errors,
    std::vector<double>& model_coeffs) {
    
    spdlog::info("开始使用ALGLIB RBF模型拟合二维样条...");
    
    if (recalls.size() != coverages.size() || recalls.size() != errors.size() || recalls.empty()) {
        spdlog::error("输入数据维度不匹配或为空");
        return FAILURE;
    }
    
    try {
        // 1. 先复制原始数据并确保单调性
        std::vector<ERROR_TYPE> filtered_recalls = recalls;
        std::vector<ERROR_TYPE> filtered_coverages = coverages;
        std::vector<ERROR_TYPE> filtered_errors = errors;
        filter_monotonic_data(filtered_recalls, filtered_coverages, filtered_errors);
        
        // 2. 准备数据点
        const size_t n_points = filtered_recalls.size();
        
        // 3. 计算边界值（用于存储模型参数）
        double min_recall = *std::min_element(filtered_recalls.begin(), filtered_recalls.end());
        double max_recall = *std::max_element(filtered_recalls.begin(), filtered_recalls.end());
        double min_coverage = *std::min_element(filtered_coverages.begin(), filtered_coverages.end());
        double max_coverage = *std::max_element(filtered_coverages.begin(), filtered_coverages.end());
        
        // 4. 创建ALGLIB数据结构
        alglib::real_2d_array xy;
        xy.setlength(n_points, 3); // 3列: recall, coverage, error
        
        // 5. 填充数据
        for (size_t i = 0; i < n_points; i++) {
            xy[i][0] = filtered_recalls[i];
            xy[i][1] = filtered_coverages[i];
            xy[i][2] = filtered_errors[i];
        }
        
        // 6. 创建RBF模型并设置
        alglib::rbfmodel model;
        alglib::rbfreport rep;
        
        // 7. 设置模型参数 - 使用薄板样条作为径向基函数
        const double radius = 1.0;      // 影响平滑度的半径参数
        const double smoothing = 0.1;   // 平滑参数 (0.0表示完全插值)
        
        // 8. 构建和训练模型 - 注意参数顺序变更!
        alglib::rbfcreate(2, 1, model);           // 2D输入，1D输出
        alglib::rbfsetpoints(model, xy);          // 设置数据点
        alglib::rbfsetalgothinplatespline(model, radius); // 设置算法
        // 注意这里改变了参数顺序：model, rep, smoothing
        alglib::rbfbuildmodel(model, rep); // 没有smoothing参数，或作为可选的xparams
        
        // 9. 检查训练结果
        if (rep.terminationtype < 0) {
            spdlog::error("ALGLIB RBF模型拟合失败: 终止类型={}", rep.terminationtype);
            return FAILURE;
        }
        
        // 10. 使用新的字符串序列化API
        std::string serialized_model;
        alglib::rbfserialize(model, serialized_model);
        
        // 11. 存储基本参数
        model_coeffs.clear();
        model_coeffs.push_back(min_recall);
        model_coeffs.push_back(max_recall);
        model_coeffs.push_back(min_coverage);
        model_coeffs.push_back(max_coverage);
        model_coeffs.push_back(static_cast<double>(serialized_model.size())); // 字符串长度
        
        // 12. 将序列化字符串存储为double数组
        for (char c : serialized_model) {
            model_coeffs.push_back(static_cast<double>(static_cast<unsigned char>(c)));
        }
        
        // 13. 更新回归系数并标记模型已拟合
        regression_coeffs_ = model_coeffs;
        is_fitted_ = true;
        
        spdlog::info("ALGLIB RBF模型创建成功！存储了{}个参数，序列化长度: {}", 
                    model_coeffs.size(), serialized_model.size());
        return SUCCESS;
    }
    catch (alglib::ap_error& e) {
        spdlog::error("ALGLIB RBF模型拟合失败: {}", e.msg);
        return FAILURE;
    }
    catch (const std::exception& e) {
        spdlog::error("RBF模型拟合失败: {}", e.what());
        return FAILURE;
    }
}



// ALGLIB二维样条预测函数 - 使用RBF模型
ERROR_TYPE upcite::ConformalRegressor::predict_alglib_quadratic_spline(
    ERROR_TYPE recall, 
    ERROR_TYPE coverage, 
    const std::vector<double>& model_coeffs) const {
    
    if (!is_fitted_ || model_coeffs.empty()) {
        spdlog::error("模型未拟合或模型系数为空");
        return 0.0;
    }
    
    // 至少需要5个基本参数
    if (model_coeffs.size() < 5) {
        spdlog::error("模型系数不完整，无法预测");
        return 0.0;
    }
    
    try {
        // 1. 提取基本参数
        double min_recall = model_coeffs[0];
        double max_recall = model_coeffs[1];
        double min_coverage = model_coeffs[2];
        double max_coverage = model_coeffs[3];
        size_t str_len = static_cast<size_t>(model_coeffs[4]);
        
        // 2. 处理超出范围的输入值
        ERROR_TYPE bounded_recall = std::max(min_recall, std::min(max_recall, recall));
        ERROR_TYPE bounded_coverage = std::max(min_coverage, std::min(max_coverage, coverage));
        
        // 3. 检查是否有足够的系数
        if (model_coeffs.size() < 5 + str_len) {
            spdlog::error("模型系数大小不足: 需要{}，实际{}", 5 + str_len, model_coeffs.size());
            return 0.0;
        }
        
        // 4. 从model_coeffs重建序列化字符串
        std::string serialized_model;
        serialized_model.reserve(str_len);
        for (size_t i = 0; i < str_len; i++) {
            serialized_model.push_back(static_cast<char>(static_cast<unsigned char>(model_coeffs[5 + i])));
        }
        
        // 5. 反序列化恢复模型
        alglib::rbfmodel model;
        alglib::rbfunserialize(serialized_model, model);
        
        // 6. 使用模型预测
        double predicted_error = alglib::rbfcalc2(model, bounded_recall, bounded_coverage);
        // 检查预测值，如果为负数则转为0
        if (predicted_error < 0) {
            printf("\n原始predicted_error为负值: %.4f，已调整为0", predicted_error);
            spdlog::warn("RBF预测出负值 {:.4f} 在 (R={:.2f}, C={:.2f})，已调整为0", 
                        predicted_error, bounded_recall, bounded_coverage);
            predicted_error = 0.0;
        } else {
            printf("\npredicted_error: %.4f", predicted_error);
        }
        // 7. 添加优化因子
        double base_alpha = 0.2;         // 基础误差值
        double correction_factor = 2.5;  // 校正因子
        
        // 8. 最终预测值
        double final_prediction = base_alpha + predicted_error * correction_factor;
        // printf("\npredicted_error: %.4f", predicted_error);
        printf("\nfinal_prediction: %.4f\n", final_prediction);
        // 9. 确保预测值为正
        return std::max(0.2, final_prediction);
    }
    catch (alglib::ap_error& e) {
        spdlog::error("ALGLIB RBF预测失败: {}", e.msg);
        
        // 失败时回退到简单的Shepard插值
        return fallback_predict(recall, coverage, model_coeffs);
    }
    catch (const std::exception& e) {
        spdlog::error("RBF预测失败: {}", e.what());
        return fallback_predict(recall, coverage, model_coeffs);
    }
}




// 添加这个函数的实现
ERROR_TYPE upcite::ConformalRegressor::fallback_predict(
    ERROR_TYPE recall, 
    ERROR_TYPE coverage, 
    const std::vector<double>& model_coeffs) const {
    printf("============开始使用简单的常数回退预测==============\n");
    // 提取基本参数
    double min_recall = model_coeffs[0];
    double max_recall = model_coeffs[1];
    double min_coverage = model_coeffs[2];
    double max_coverage = model_coeffs[3];
    
    // 处理超出范围的输入值
    ERROR_TYPE bounded_recall = std::max(min_recall, std::min(max_recall, recall));
    ERROR_TYPE bounded_coverage = std::max(min_coverage, std::min(max_coverage, coverage));
    
    // 使用简单的常数回退
    return 0.5; // 一个安全的默认值
}


// 预测Eigen二次样条模型
ERROR_TYPE upcite::ConformalRegressor::predict_eigen_quadratic_spline_old(
    ERROR_TYPE recall, 
    ERROR_TYPE coverage, 
    const std::vector<double>& coeffs) const {
    if (coeffs.size() < 8) {
        spdlog::error("模型系数不足");
        return 0.0;
    }
    // 解析模型参数
    double min_recall = coeffs[0];
    double max_recall = coeffs[1];
    double min_coverage = coeffs[2];
    double max_coverage = coeffs[3];
    int recall_points = static_cast<int>(coeffs[4]);
    int coverage_points = static_cast<int>(coeffs[5]);
    int spline_degree = static_cast<int>(coeffs[6]);
    double smoothing = coeffs[7];
    
    // 确保recall和coverage在范围内
    if (recall < min_recall) recall = min_recall;
    if (recall > max_recall) recall = max_recall;
    if (coverage < min_coverage) coverage = min_coverage;
    if (coverage > max_coverage) coverage = max_coverage;
    
    // 从系数中提取原始点
    std::vector<double> unique_recalls;
    std::vector<double> unique_coverages;
    
    int offset = 8;
    for (int i = 0; i < recall_points; i++) {
        unique_recalls.push_back(coeffs[offset + i]);
    }
    
    offset += recall_points;
    for (int i = 0; i < coverage_points; i++) {
        unique_coverages.push_back(coeffs[offset + i]);
    }
    
    // 找到最近的点进行插值
    // 找到recall的位置
    int r_idx = 0;
    for (int i = 0; i < recall_points; i++) {
        if (unique_recalls[i] > recall) {
            break;
        }
        r_idx = i;
    }
    
    // 找到coverage的位置
    int c_idx = 0;
    for (int i = 0; i < coverage_points; i++) {
        if (unique_coverages[i] > coverage) {
            break;
        }
        c_idx = i;
    }
    
    // 网格数据起始位置
    offset += coverage_points;
    
    // 提取周围的数据点
    double z00 = 0.0, z01 = 0.0, z10 = 0.0, z11 = 0.0;
    double r0 = 0.0, r1 = 0.0, c0 = 0.0, c1 = 0.0;
    
    r0 = unique_recalls[r_idx];
    c0 = unique_coverages[c_idx];
    
    // 获取对应网格中的值
    int grid_idx = c_idx * recall_points + r_idx;
    z00 = coeffs[offset + grid_idx];
    
    // 处理边界情况
    if (r_idx + 1 < recall_points) {
        r1 = unique_recalls[r_idx + 1];
        z10 = coeffs[offset + c_idx * recall_points + (r_idx + 1)];
    } else {
        r1 = r0;
        z10 = z00;
    }
    
    if (c_idx + 1 < coverage_points) {
        c1 = unique_coverages[c_idx + 1];
        z01 = coeffs[offset + (c_idx + 1) * recall_points + r_idx];
    } else {
        c1 = c0;
        z01 = z00;
    }
    
    if (r_idx + 1 < recall_points && c_idx + 1 < coverage_points) {
        z11 = coeffs[offset + (c_idx + 1) * recall_points + (r_idx + 1)];
    } else {
        z11 = z00;
    }
    
    // 双线性插值
    double t_r = 0.0, t_c = 0.0;
    if (r1 != r0) t_r = (recall - r0) / (r1 - r0);
    if (c1 != c0) t_c = (coverage - c0) / (c1 - c0);
    
    // 执行插值
    double z0 = z00 * (1 - t_r) + z10 * t_r;
    double z1 = z01 * (1 - t_r) + z11 * t_r;
    double z = z0 * (1 - t_c) + z1 * t_c;
    
    // 添加Python RectBivariateSpline的校正因子，使值更接近真实值
    // 根据测试，1.2的系数可以使得预测值更接近Python的结果
    double correction_factor =2.5;//1.2
    // 添加底层偏置
    double base_alpha = 0.4; // 基于Q1值(0.316~0.418)设置合理的基础alpha值
    return z * correction_factor + base_alpha;
}





// 原始Eigen二次样条模型的相关函数
// 实现Eigen二次样条模型 (k=2, s=3.0) - 类似Python的RectBivariateSpline
RESPONSE upcite::ConformalRegressor::fit_eigen_quadratic_spline_old(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages, 
    const std::vector<ERROR_TYPE>& errors,
    std::vector<double>& model_coeffs) {
    
    spdlog::info("开始使用Eigen库拟合二次样条模型 (k=2, s=3.0)...");
    // printf("开始使用Eigen库拟合二次样条模型 (k=2, s=3.0)...\n");
    
    if (recalls.size() != coverages.size() || recalls.size() != errors.size() || recalls.empty()) {
        spdlog::error("输入数据维度不匹配或为空");
        printf("输入数据维度不匹配或为空\n");
        return FAILURE;
    }
    // 先复制原始数据
    std::vector<ERROR_TYPE> filtered_recalls = recalls;
    std::vector<ERROR_TYPE> filtered_coverages = coverages;
    std::vector<ERROR_TYPE> filtered_errors = errors;
    // 只传递3个参数
    // filter_monotonic_data(filtered_recalls, filtered_coverages, filtered_errors);
    // printf("Eigen样条模型: 数据过滤前 %zu 个点, 过滤后 %zu 个点\n", 
    //        recalls.size(), filtered_recalls.size());
    // 使用过滤后的数据进行拟合
    // 后续拟合逻辑中使用filtered_recalls, filtered_coverages, filtered_errors代替原始数据
    // 获取唯一的recall和coverage值
    std::vector<ERROR_TYPE> unique_recalls = filtered_recalls;
    std::vector<ERROR_TYPE> unique_coverages = filtered_coverages;
    // 去重并排序
    std::sort(unique_recalls.begin(), unique_recalls.end());
    unique_recalls.erase(std::unique(unique_recalls.begin(), unique_recalls.end()), unique_recalls.end());
    
    std::sort(unique_coverages.begin(), unique_coverages.end());
    unique_coverages.erase(std::unique(unique_coverages.begin(), unique_coverages.end()), unique_coverages.end());
    
    // spdlog::info("唯一recall值数量: {}", unique_recalls.size());
    // spdlog::info("唯一coverage值数量: {}", unique_coverages.size());
    // printf("唯一recall值数量: %zu\n", unique_recalls.size());
    // printf("唯一coverage值数量: %zu\n", unique_coverages.size());
    
    // 创建网格数据
    Eigen::MatrixXd grid_values(unique_coverages.size(), unique_recalls.size());
    
    // 填充网格数据
    for (size_t i = 0; i < unique_coverages.size(); i++) {
        for (size_t j = 0; j < unique_recalls.size(); j++) {
            ERROR_TYPE cov = unique_coverages[i];
            ERROR_TYPE rec = unique_recalls[j];
            
            // 找到最接近当前网格点的数据点
            size_t closest_idx = 0;
            ERROR_TYPE min_dist = std::numeric_limits<ERROR_TYPE>::max();
            
            for (size_t k = 0; k < filtered_recalls.size(); k++) {
                ERROR_TYPE dist = std::pow(filtered_recalls[k] - rec, 2) + std::pow(filtered_coverages[k] - cov, 2);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_idx = k;
                }
            }
            
            grid_values(i, j) = filtered_errors[closest_idx];
        }
    }
    
    // 创建用于样条拟合的数据结构
    int spline_degree = 2;  // 二次样条
    double smoothing = 3.0; // 平滑因子
    
    // 存储样条系数
    try {
        // 数据映射到[0,1]范围以提高数值稳定性
        double min_recall = unique_recalls.front();
        double max_recall = unique_recalls.back();
        double min_coverage = unique_coverages.front();
        double max_coverage = unique_coverages.back();
        
        // 将关键参数存储在coeffs中
        model_coeffs.clear();
        
        // 存储recall和coverage范围
        model_coeffs.push_back(min_recall);
        model_coeffs.push_back(max_recall);
        model_coeffs.push_back(min_coverage);
        model_coeffs.push_back(max_coverage);
        
        // 存储唯一点数量
        model_coeffs.push_back(unique_recalls.size());
        model_coeffs.push_back(unique_coverages.size());
        
        // 存储样条阶数和平滑参数
        model_coeffs.push_back(spline_degree);
        model_coeffs.push_back(smoothing);
        
        // 存储数据点 - 先是recall点
        for (double val : unique_recalls) {
            model_coeffs.push_back(val);
        }
        
        // 存储coverage点
        for (double val : unique_coverages) {
            model_coeffs.push_back(val);
        }
        
        // 存储网格数据
        for (size_t i = 0; i < unique_coverages.size(); i++) {
            for (size_t j = 0; j < unique_recalls.size(); j++) {
                model_coeffs.push_back(grid_values(i, j));
            }
        }
        
        // 更新回归系数，供预测使用
        regression_coeffs_ = model_coeffs;
        
        spdlog::info("Eigen二次样条模型 (k=2, s=3.0) 创建成功！存储了 {} 个参数", model_coeffs.size());
        // printf("Eigen二次样条模型 (k=2, s=3.0) 创建成功！存储了 %zu 个参数\n", model_coeffs.size());
        
        // 标记模型已拟合
        is_fitted_ = true;
        
        return SUCCESS;
    }
    catch (const std::exception& e) {
        spdlog::error("Eigen样条拟合失败: {}", e.what());
        printf("Eigen样条拟合失败: %s\n", e.what());
        return FAILURE;
    }
}






// Add new functions after the fit_batch_bivariate_regression function

RESPONSE upcite::ConformalRegressor::filter_monotonic_data(
    std::vector<ERROR_TYPE>& recalls,
    std::vector<ERROR_TYPE>& coverages,
    std::vector<ERROR_TYPE>& errors) {
    
    // printf("开始优化数据以确保单调性约束，优先保留误差大的点...\n");
    // 记录初始数据点数量
    size_t initial_points = recalls.size();
    if (initial_points == 0) {
        spdlog::error("没有数据点可供处理");
        return FAILURE;
    }
    
    // 打印过滤前的所有数据点
    // printf("======== 过滤前的数据点（共 %zu 个）========\n", initial_points);
    // printf("%-5s %-10s %-10s %-15s\n", "索引", "召回率", "覆盖率", "误差值");
    // for (size_t i = 0; i < initial_points; i++) {
    //     printf("%-5zu %-10.6f %-10.6f %-15.6f\n", i, recalls[i], coverages[i], errors[i]);
    // }

    // std::vector<std::pair<double, double>> key_points = {
    //   {0.9, 0.9}, {0.9, 0.95}, {0.9, 0.99},
    //   {0.95, 0.9}, {0.95, 0.95}, {0.95, 0.99},
    //   {0.99, 0.9}, {0.99, 0.95}, {0.99, 0.99}
    // };

    // printf("======== 关键点真实数据（共 %zu 个关键点）========\n", key_points.size());
    // printf("%-10s %-10s %-15s %-15s\n", "actual_r", "actual_c", "error", "distance");

    // for (const auto& key_point : key_points) {
    //   double target_recall = key_point.first;
    //   double target_coverage = key_point.second;
      
    //   // 寻找精确匹配或最近点
    //   bool found_exact = false;
    //   size_t best_idx = 0;
    //   double min_distance = std::numeric_limits<double>::max();
      
    //   for (size_t i = 0; i < initial_points; i++) {
    //     if (std::abs(recalls[i] - target_recall) < 1e-6 && 
    //         std::abs(coverages[i] - target_coverage) < 1e-6) {
    //       found_exact = true;
    //       best_idx = i;
    //       break;
    //     }
        
    //     double distance = std::sqrt(std::pow(recalls[i] - target_recall, 2) + 
    //                               std::pow(coverages[i] - target_coverage, 2));
    //     if (distance < min_distance) {
    //       min_distance = distance;
    //       best_idx = i;
    //     }
    //   }
      
    //   // 打印结果
    //   if (found_exact) {
    //     printf("%-10.2f %-10.2f %-15.4f %-15s\n", 
    //           recalls[best_idx], coverages[best_idx], errors[best_idx], "精确匹配");
    //   } else {
    //     printf("%-10.2f %-10.2f %-15.4f %-15.6f\n", 
    //           recalls[best_idx], coverages[best_idx], errors[best_idx], min_distance);
    //   }
    // }
    
    // 复制原始数据，用于后续比较
    std::vector<ERROR_TYPE> original_recalls = recalls;
    std::vector<ERROR_TYPE> original_coverages = coverages;
    std::vector<ERROR_TYPE> original_errors = errors;
    
    // 创建数据点索引向量，用于后续处理
    std::vector<size_t> indices(initial_points);
    for (size_t i = 0; i < initial_points; i++) {
        indices[i] = i;
    }
    
    // 1. 异常值处理（修改为保留高误差值）
    // 计算误差的四分位数范围(IQR)，但只用于移除违反单调性且误差较小的点
    std::vector<ERROR_TYPE> sorted_errors = errors;
    std::sort(sorted_errors.begin(), sorted_errors.end());
    
    // 获取最小和最大误差值
    ERROR_TYPE min_error = sorted_errors.front(); // 最小值
    ERROR_TYPE max_error = sorted_errors.back();  // 最大值

    size_t q1_idx = sorted_errors.size() / 4;
    size_t q3_idx = sorted_errors.size() * 3 / 4;
    
    ERROR_TYPE q1 = sorted_errors[q1_idx];
    ERROR_TYPE q3 = sorted_errors[q3_idx];
    ERROR_TYPE iqr = q3 - q1;
    ERROR_TYPE lower_bound = q1;  // 定义下界异常值
    
    // printf("Q1=%.3f, Q3=%.3f,  lower_bound=%.3f, max_error=%.3f, min_error=%.3f\n", 
    //        q1, q3, lower_bound, max_error, min_error);
    
    std::vector<size_t> outlier_indices;
    for (size_t i = 0; i < errors.size(); i++) {
        if (errors[i] < lower_bound) {  // 寻找误差较小的异常值
            // 检查是否违反单调性
            bool is_violating = false;
            for (size_t j = 0; j < errors.size(); j++) {
                if (i == j) continue;
                
                // 如果存在recall和coverage都较大但error较小的点
                if (recalls[j] >= recalls[i] && coverages[j] >= coverages[i] && 
                    (recalls[j] > recalls[i] || coverages[j] > coverages[i]) && 
                    errors[j] <= errors[i]) {
                    is_violating = true;
                    break;
                }
            }
            
            if (is_violating) {
                outlier_indices.push_back(i);
            }
        }
    }
    
    // 应用异常值过滤
    if (!outlier_indices.empty()) {
        // printf("将移除 %zu 个违反单调性的小误差异常点\n", outlier_indices.size());
        
        // 打印将被移除的点
        // printf("\n将移除的异常点:\n");
        // printf("%-5s %-10s %-10s %-15s\n", "索引", "召回率", "覆盖率", "误差值");
        // for (size_t idx : outlier_indices) {
        //     printf("%-5zu %-10.6f %-10.6f %-15.6f\n", idx, recalls[idx], coverages[idx], errors[idx]);
        // }
        
        // 标记需要移除的索引
        std::vector<bool> to_remove(initial_points, false);
        for (size_t idx : outlier_indices) {
            to_remove[idx] = true;
        }
        
        // 创建过滤后的数据
        std::vector<ERROR_TYPE> filtered_recalls, filtered_coverages, filtered_errors;
        for (size_t i = 0; i < initial_points; i++) {
            if (!to_remove[i]) {
                filtered_recalls.push_back(recalls[i]);
                filtered_coverages.push_back(coverages[i]);
                filtered_errors.push_back(errors[i]);
            }
        }
        
        recalls = filtered_recalls;
        coverages = filtered_coverages;
        errors = filtered_errors;
        
        // printf("异常值过滤后剩余 %zu 个点\n", recalls.size());
    }
    
    // 2. 按照recall分组，确保coverage增加时error也增加
    // printf("应用修改后的约束1: 当recall固定时，coverage增加，error应该增加\n");
    
    // 创建recall分组
    std::map<ERROR_TYPE, std::vector<size_t>> recall_groups;
    for (size_t i = 0; i < recalls.size(); i++) {
        recall_groups[recalls[i]].push_back(i);
    }
    
    // 跟踪需要调整的点
    std::vector<std::tuple<size_t, ERROR_TYPE>> adjustments; // (索引, 新误差值)
    
    for (auto& group : recall_groups) {
        std::vector<size_t>& indices = group.second;
        if (indices.size() <= 1) continue; // 单点不需处理
        
        // 按coverage排序
        std::sort(indices.begin(), indices.end(), 
            [&coverages](size_t a, size_t b) { return coverages[a] < coverages[b]; });
        
        // 遍历排序后的点，确保单调性，优先保留高误差点
        for (size_t i = 0; i < indices.size() - 1; i++) {
            size_t curr_idx = indices[i];
            size_t next_idx = indices[i+1];
            
            // 如果coverage增加但error没有增加
            if (coverages[next_idx] > coverages[curr_idx] && 
                errors[next_idx] <= errors[curr_idx]) {
                
                // 保留高误差值，调整低误差点
                if (errors[next_idx] < errors[curr_idx]) {
                    // 增加next_idx的误差值，使其略高于curr_idx
                    adjustments.push_back(std::make_tuple(
                        next_idx, errors[curr_idx] * 1.01)); // 增加1%
                } else if (errors[next_idx] == errors[curr_idx]) {
                    // 如果相等，略微增加next_idx的误差
                    adjustments.push_back(std::make_tuple(
                        next_idx, errors[curr_idx] * 1.01));
                }
            }
        }
    }
    
    // // 打印约束1的调整
    // if (!adjustments.empty()) {
    //     printf("\n约束1的调整点 (共 %zu 个):\n", adjustments.size());
    //     printf("%-5s %-10s %-10s %-15s %-15s\n", "索引", "召回率", "覆盖率", "原误差", "新误差");
    //     for (auto& adj : adjustments) {
    //         size_t idx;
    //         ERROR_TYPE new_error;
    //         std::tie(idx, new_error) = adj;
    //         printf("%-5zu %-10.6f %-10.6f %-15.6f %-15.6f\n", 
    //               idx, recalls[idx], coverages[idx], errors[idx], new_error);
    //     }
    // }
    
    // 应用误差调整
    for (auto& adj : adjustments) {
        size_t idx;
        ERROR_TYPE new_error;
        std::tie(idx, new_error) = adj;
        errors[idx] = new_error;
    }
    
    // printf("约束1：调整了 %zu 个点的误差值\n", adjustments.size());
    
    // 3. 按照coverage分组，确保recall增加时error也增加
    // printf("应用修改后的约束2: 当coverage固定时，recall增加，error应该增加\n");
    
    // 创建coverage分组
    std::map<ERROR_TYPE, std::vector<size_t>> coverage_groups;
    for (size_t i = 0; i < coverages.size(); i++) {
        coverage_groups[coverages[i]].push_back(i);
    }
    
    // 清空调整列表，准备新的调整
    adjustments.clear();
    
    for (auto& group : coverage_groups) {
        std::vector<size_t>& indices = group.second;
        if (indices.size() <= 1) continue; // 单点不需处理
        
        // 按recall排序
        std::sort(indices.begin(), indices.end(), 
            [&recalls](size_t a, size_t b) { return recalls[a] < recalls[b]; });
        
        // 遍历排序后的点，确保单调性，优先保留高误差点
        for (size_t i = 0; i < indices.size() - 1; i++) {
            size_t curr_idx = indices[i];
            size_t next_idx = indices[i+1];
            
            // 如果recall增加但error没有增加
            if (recalls[next_idx] > recalls[curr_idx] && 
                errors[next_idx] <= errors[curr_idx]) {
                
                // 保留高误差值，调整低误差点
                if (errors[next_idx] < errors[curr_idx]) {
                    // 增加next_idx的误差值，使其略高于curr_idx
                    adjustments.push_back(std::make_tuple(
                        next_idx, errors[curr_idx] * 1.01)); // 增加1%
                } else if (errors[next_idx] == errors[curr_idx]) {
                    // 如果相等，略微增加next_idx的误差
                    adjustments.push_back(std::make_tuple(
                        next_idx, errors[curr_idx] * 1.01));
                }
            }
        }
    }
    
    // // 打印约束2的调整
    // if (!adjustments.empty()) {
    //     printf("\n约束2的调整点 (共 %zu 个):\n", adjustments.size());
    //     printf("%-5s %-10s %-10s %-15s %-15s\n", "索引", "召回率", "覆盖率", "原误差", "新误差");
    //     for (auto& adj : adjustments) {
    //         size_t idx;
    //         ERROR_TYPE new_error;
    //         std::tie(idx, new_error) = adj;
    //         printf("%-5zu %-10.6f %-10.6f %-15.6f %-15.6f\n", 
    //               idx, recalls[idx], coverages[idx], errors[idx], new_error);
    //     }
    // }
    
    // 应用误差调整
    for (auto& adj : adjustments) {
        size_t idx;
        ERROR_TYPE new_error;
        std::tie(idx, new_error) = adj;
        errors[idx] = new_error;
    }
    
    // printf("约束2：调整了 %zu 个点的误差值\n", adjustments.size());
    
    // 4. 确保当recall和coverage都增加时，error也增加
    // printf("应用修改后的约束3: 当recall和coverage都增加时，error应该增加\n");
    
    // 清空调整列表，准备新的调整
    adjustments.clear();
    
    for (size_t i = 0; i < recalls.size(); i++) {
        for (size_t j = 0; j < recalls.size(); j++) {
            if (i == j) continue;
            
            // 如果点j的recall和coverage都大于点i
            if (recalls[j] > recalls[i] && coverages[j] > coverages[i]) {
                // error也应该更大
                if (errors[j] <= errors[i]) {
                    // 调整j点，使其误差略高于i点
                    adjustments.push_back(std::make_tuple(
                        j, errors[i] * 1.01)); // 增加1%
                }
            }
        }
    }
    
    // // 打印约束3的调整
    // if (!adjustments.empty()) {
    //     printf("\n约束3的调整点 (共 %zu 个):\n", adjustments.size());
    //     printf("%-5s %-10s %-10s %-15s %-15s\n", "索引", "召回率", "覆盖率", "原误差", "新误差");
    //     for (auto& adj : adjustments) {
    //         size_t idx;
    //         ERROR_TYPE new_error;
    //         std::tie(idx, new_error) = adj;
    //         printf("%-5zu %-10.6f %-10.6f %-15.6f %-15.6f\n", 
    //               idx, recalls[idx], coverages[idx], errors[idx], new_error);
    //     }
    // }
    
    // 应用误差调整
    for (auto& adj : adjustments) {
        size_t idx;
        ERROR_TYPE new_error;
        std::tie(idx, new_error) = adj;
        errors[idx] = new_error;
    }
    
    // printf("约束3：调整了 %zu 个点的误差值\n", adjustments.size());
    
    // 打印过滤和调整后的数据点
    // printf("======== 过滤和调整后的数据点（共 %zu 个）========\n", recalls.size());
    // printf("%-5s %-10s %-10s %-15s %-15s %-10s\n", "索引", "召回率", "覆盖率", "原误差", "新误差", "状态");
    // 创建索引映射，用于追踪原始点的去向
    std::map<std::pair<ERROR_TYPE, ERROR_TYPE>, size_t> point_to_original_idx;
    for (size_t i = 0; i < original_recalls.size(); i++) {
        point_to_original_idx[{original_recalls[i], original_coverages[i]}] = i;
    }
    
    // // 创建一个集合，用于记录已输出的原始点索引
    // std::set<size_t> processed_original_indices;
    // // 输出保留或修改的点
    // for (size_t i = 0; i < recalls.size(); i++) {
    //     std::pair<ERROR_TYPE, ERROR_TYPE> point_key = {recalls[i], coverages[i]};
    //     size_t orig_idx = point_to_original_idx[point_key];
    //     processed_original_indices.insert(orig_idx);
    //     std::string status = "保留";
    //     if (std::abs(errors[i] - original_errors[orig_idx]) > 1e-6) {
    //         status = "修改";
    //     }
    //     // printf("%-5zu %-10.6f %-10.6f %-15.6f %-15.6f %-10s\n", orig_idx, recalls[i], coverages[i], original_errors[orig_idx], errors[i], status.c_str());
    // }
    
    // // 输出被删除的点
    // for (size_t i = 0; i < original_recalls.size(); i++) {
    //     if (processed_original_indices.find(i) == processed_original_indices.end()) {
    //         printf("%-5zu %-10.6f %-10.6f %-15.6f %-15s %-10s\n", 
    //               i, original_recalls[i], original_coverages[i], original_errors[i], "-", "删除");
    //     }
    // }
    
    // 检查过滤前后数据点数量的变化
    // printf("过滤前: %zu, 过滤后: %zu\n", initial_points, recalls.size());
    // printf("过滤后数据点数量: %zu\n", recalls.size());
    // printf("删除的点数量: %zu\n", initial_points - recalls.size());
    // printf("修改的点数量: %zu\n", adjustments.size());
    
    // 最后查重并汇总报告
    // printf("\n单调性处理完成，总共调整了 %zu 个点的误差值\n", 
    //        adjustments.size() + outlier_indices.size());
    
    return SUCCESS;
}





// RESPONSE upcite::ConformalRegressor::fit_optimized_regional_spline(
//     std::vector<ERROR_TYPE>& recalls,
//     std::vector<ERROR_TYPE>& coverages, 
//     std::vector<ERROR_TYPE>& errors,
//     ERROR_TYPE high_recall_threshold,
//     ERROR_TYPE min_coverage_threshold) {
    
//     spdlog::info("开始拟合优化区域样条模型");
//     printf("开始拟合优化区域样条模型...\n");
    
//     // 1. 首先过滤数据，确保单调性
//     RESPONSE filter_result = filter_monotonic_data(recalls, coverages, errors);
//     if (filter_result != SUCCESS) {
//         spdlog::error("数据过滤失败");
//         return FAILURE;
//     }
    
//     // 2. 准备网格数据
//     // 2.1 获取唯一的recall和coverage值
//     std::vector<ERROR_TYPE> unique_recalls = recalls;
//     std::vector<ERROR_TYPE> unique_coverages = coverages;
    
//     // 去重
//     std::sort(unique_recalls.begin(), unique_recalls.end());
//     unique_recalls.erase(std::unique(unique_recalls.begin(), unique_recalls.end()), unique_recalls.end());
    
//     std::sort(unique_coverages.begin(), unique_coverages.end());
//     unique_coverages.erase(std::unique(unique_coverages.begin(), unique_coverages.end()), unique_coverages.end());
    
//     printf("唯一recall值: %zu个\n", unique_recalls.size());
//     printf("唯一coverage值: %zu个\n", unique_coverages.size());
    
//     // 2.2 创建原始网格数据
//     std::vector<std::vector<ERROR_TYPE>> grid_values(unique_coverages.size(), 
//                                                    std::vector<ERROR_TYPE>(unique_recalls.size(), 0));
    
//     // 2.3 填充网格数据
//     for (size_t i = 0; i < unique_coverages.size(); i++) {
//         for (size_t j = 0; j < unique_recalls.size(); j++) {
//             ERROR_TYPE cov = unique_coverages[i];
//             ERROR_TYPE rec = unique_recalls[j];
            
//             // 找到最接近当前网格点的数据点
//             size_t closest_idx = 0;
//             ERROR_TYPE min_dist = constant::MAX_VALUE;
            
//             for (size_t k = 0; k < recalls.size(); k++) {
//                 ERROR_TYPE dist = std::pow(recalls[k] - rec, 2) + std::pow(coverages[k] - cov, 2);
//                 if (dist < min_dist) {
//                     min_dist = dist;
//                     closest_idx = k;
//                 }
//             }
            
//             grid_values[i][j] = errors[closest_idx];
//         }
//     }
    
//     // 3. 区域优化：在高召回率区域应用单调性约束
//     // 3.1 标识高召回率区域
//     std::vector<size_t> high_recall_indices;
//     for (size_t j = 0; j < unique_recalls.size(); j++) {
//         if (unique_recalls[j] >= high_recall_threshold) {
//             high_recall_indices.push_back(j);
//         }
//     }
    
//     printf("\n高recall区域 (≥%.2f) 包含 %zu 个recall值\n", 
//            high_recall_threshold, high_recall_indices.size());
    
//     // 3.2 创建区域优化网格
//     std::vector<std::vector<ERROR_TYPE>> regional_grid = grid_values;
    
//     // 3.3 仅在高召回率区域应用单调性约束
//     for (size_t j : high_recall_indices) {
//         // 获取当前召回率的列值
//         std::vector<ERROR_TYPE> col_values = regional_grid[0]; // 初始化为第一行的值
//         for (size_t i = 0; i < unique_coverages.size(); i++) {
//             col_values[i] = regional_grid[i][j];
//         }
        
//         // 仅在覆盖率 >= min_coverage_threshold 的区域应用约束
//         std::vector<size_t> cov_indices;
//         for (size_t i = 0; i < unique_coverages.size(); i++) {
//             if (unique_coverages[i] >= min_coverage_threshold) {
//                 cov_indices.push_back(i);
//             }
//         }
        
//         // 确保单调增加
//         for (size_t idx = 1; idx < cov_indices.size(); idx++) {
//             size_t i = cov_indices[idx];
//             size_t i_prev = cov_indices[idx-1];
//             if (col_values[i] < col_values[i_prev]) {
//                 col_values[i] = col_values[i_prev];
//             }
//         }
        
//         // 更新网格
//         for (size_t i = 0; i < unique_coverages.size(); i++) {
//             regional_grid[i][j] = col_values[i];
//         }
//     }
    
//     // 4. 使用优化后的网格创建样条模型
//     // 4.1 将二维网格转换为一维训练数据
//     std::vector<ERROR_TYPE> train_recalls;
//     std::vector<ERROR_TYPE> train_coverages;
//     std::vector<ERROR_TYPE> train_errors;
    
//     for (size_t i = 0; i < unique_coverages.size(); i++) {
//         for (size_t j = 0; j < unique_recalls.size(); j++) {
//             train_recalls.push_back(unique_recalls[j]);
//             train_coverages.push_back(unique_coverages[i]);
//             train_errors.push_back(regional_grid[i][j]);
//         }
//     }
    
//     // 创建二次多项式模型 (k=2)
//     printf("创建区域优化的二次样条模型...\n");
//     RESPONSE fit_result = train_optimal_polynomial_model(
//         train_recalls, train_coverages, train_errors, 2);
    
//     if (fit_result != SUCCESS) {
//         spdlog::error("拟合二次多项式模型失败");
//         return FAILURE;
//     }
    
//     printf("区域优化样条模型创建成功！\n");
//     return SUCCESS;
// }



// // Add a convenience method to predict error value using optimized regional spline model
// ERROR_TYPE upcite::ConformalRegressor::predict_regional_spline_error(
//     ERROR_TYPE recall, ERROR_TYPE coverage) const {
//     // Call the existing predict_polynomial_model_error with the appropriate parameters
//     // Assuming we're using a quadratic model (degree=2)
//     return predict_polynomial_model_error(recall, coverage, regression_coeffs_, 2);
//   }
// }








// double upcite::ConformalRegressor::predict_error_value(double recall, double coverage) const {
//     if (!is_fitted_) {
//         spdlog::warn("回归模型未训练");
//         printf("回归模型未训练\n");
//         return -1.0;
//     }
    
//     // 添加spdlog日志
//     std::string coeff_str = "[";
//     for (size_t i = 0; i < std::min(regression_coeffs_.size(), size_t(10)); ++i) {
//         coeff_str += fmt::format("{:.2f}", regression_coeffs_[i]);
//         if (i < std::min(regression_coeffs_.size(), size_t(10)) - 1) {
//             coeff_str += ", ";
//         }
//     }
//     if (regression_coeffs_.size() > 10) coeff_str += ", ...";
//     coeff_str += "]";
//     spdlog::info("回归系数: {}", coeff_str);
    
//     double predicted_error = 0.0;
//     // 检查系数数量，决定使用哪种模型
//     size_t num_coeffs = regression_coeffs_.size();
    
//     // 新增：检测是否使用Eigen样条模型
//     // Eigen样条模型系数至少包含8个基础参数和额外的网格点数据
//     if (num_coeffs >= 8 && regression_coeffs_[6] >= 0) {  // coeffs[6]是spline_degree
//         // 使用Eigen二次样条模型进行预测
//         predicted_error = predict_eigen_spline_error(recall, coverage);
//         spdlog::info("使用Eigen二次样条模型预测，共{}个系数", num_coeffs);
//     }
//     else if (num_coeffs == 3) {
//         // 一阶多项式模型 (线性模型)
//         predicted_error = regression_coeffs_[0] +              // 常数项
//                          regression_coeffs_[1] * recall +     // recall的线性项
//                          regression_coeffs_[2] * coverage;    // coverage的线性项
//         spdlog::info("使用一阶多项式模型预测 (线性模型)");
//     }
//     else if (num_coeffs == 6) {
//         // 标准二次模型 (传统6参数公式)
//         predicted_error = regression_coeffs_[0] +                      // 常数项
//                          regression_coeffs_[1] * recall +             // 一次项
//                          regression_coeffs_[2] * coverage +           // 一次项
//                          regression_coeffs_[3] * recall * coverage +  // 交互项
//                          regression_coeffs_[4] * recall * recall +    // 二次项
//                          regression_coeffs_[5] * coverage * coverage; // 二次项
        
//         spdlog::info("使用标准二次模型预测");
//     }
//     else if (num_coeffs > 6) {
//         // 使用高阶多项式模型 (系数数量>6时)
//         int degree = 0;
//         // 根据系数数量确定多项式阶数
//         // 根据多项式系数公式：1 + 3 (d=1) + 4 (d=2) + 5 (d=3) ...
//         if (num_coeffs >= 16) degree = 5;
//         else if (num_coeffs >= 11) degree = 4;
//         else if (num_coeffs >= 7) degree = 3;
//         else degree = 2;
//         predicted_error = predict_polynomial_model_error(recall, coverage, regression_coeffs_, degree);
//         spdlog::info("使用{}阶多项式模型预测，共{}个系数", degree, num_coeffs);
//     }
//     else {
//         // 系数数量不匹配任何已知模型
//         spdlog::error("回归系数数量({})不支持的模型", num_coeffs);
//         printf("回归系数数量(%zu)不支持的模型\n", num_coeffs);
//         return -1.0;
//     }    
    
//     spdlog::info("预测误差值: {:.4f}, recall={:.4f}, coverage={:.4f}", 
//                 std::max(0.0, predicted_error), recall, coverage);
//     return std::max(0.0, predicted_error);
// }





// 新增函数：使用特定批次的实际误差值
// RESPONSE upcite::ConformalRegressor::train_regression_model_for_recall_coverage_actual_error(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages,
//     const std::vector<ID_TYPE>& error_indices,
//     ID_TYPE batch_id,
//     ID_TYPE filter_id) {
    
//     if (recalls.size() != coverages.size() || recalls.size() != error_indices.size() || recalls.empty()) {
//         printf("错误：输入数据数量不匹配或为空\n");
//         return FAILURE;
//     }
    
//     // 获取当前filter下的cp类的batch_alphas，用于将位置转换为实际误差值
//     const auto& batch_alphas = this->get_batch_alphas();
//     if (batch_alphas.empty()) {
//         printf("错误：batch_alphas为空，无法获取误差值\n");
//         return FAILURE;
//     }
    
//     if (batch_id >= batch_alphas.size()) {
//         printf("错误：指定的batch_id %ld 超出有效范围 [0, %zu)\n", (long)batch_id, batch_alphas.size());
//         return FAILURE;
//     }
    
//     printf("使用批次 %ld 的实际误差值训练模型，批次总数: %zu\n", (long)batch_id, batch_alphas.size());
//     spdlog::info("使用批次 {} 的实际误差值训练模型，批次总数: {}", batch_id, batch_alphas.size());
    
//     // 准备训练数据，将error_i转换为指定批次的实际误差值
//     std::vector<double> actual_errors;
//     actual_errors.reserve(error_indices.size());
    
//     for (ID_TYPE error_i : error_indices) {
//         // 从指定批次中获取实际误差值
//         if (error_i < batch_alphas[batch_id].size()) {
//             double error_value = batch_alphas[batch_id][error_i];
//             actual_errors.push_back(error_value);
//         } else {
//             printf("警告：误差位置 %ld 超出批次 %ld 的误差值范围 [0, %zu)\n", 
//                    (long)error_i, (long)batch_id, batch_alphas[batch_id].size());
//             return FAILURE;
//         }
//     }
    
//     // 打印actual_errors的大小
//     printf("过滤器 %ld 的批次 %ld 的actual_errors大小: %zu\n", 
//            (long)filter_id, (long)batch_id, actual_errors.size());
//     spdlog::info("过滤器 {} 的批次 {} 的actual_errors大小: {}", 
//                 filter_id, batch_id, actual_errors.size());
    
//     // 样本数和特征数
//     size_t n = recalls.size();  // 样本数
//     size_t p = 6;  // 特征数：常数项、recall、coverage、recall*coverage、recall^2、coverage^2
    
//     // 使用GSL库进行多元回归
//     gsl_matrix *X = gsl_matrix_alloc(n, p);
//     gsl_vector *y = gsl_vector_alloc(n);
//     gsl_vector *c = gsl_vector_alloc(p);  // 回归系数
//     gsl_matrix *cov = gsl_matrix_alloc(p, p);  // 协方差矩阵
//     double chisq;  // 拟合优度
    
//     // 填充设计矩阵X和目标向量y
//     for (size_t i = 0; i < n; i++) {
//         // 设计矩阵：[1, recall, coverage, recall*coverage, recall^2, coverage^2]
//         gsl_matrix_set(X, i, 0, 1.0);  // 常数项
//         gsl_matrix_set(X, i, 1, recalls[i]);
//         gsl_matrix_set(X, i, 2, coverages[i]);
//         gsl_matrix_set(X, i, 3, recalls[i] * coverages[i]);  // 交互项
//         gsl_matrix_set(X, i, 4, recalls[i] * recalls[i]);
//         gsl_matrix_set(X, i, 5, coverages[i] * coverages[i]);
        
//         // 目标向量 - 使用实际误差值
//         gsl_vector_set(y, i, actual_errors[i]);
//     }
    
//     // 执行回归计算
//     gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc(n, p);
//     int ret = gsl_multifit_linear(X, y, c, cov, &chisq, work);
    
//     if (ret != GSL_SUCCESS) {
//         printf("错误：GSL回归计算失败，错误码：%d\n", ret);
//         // 清理资源
//         gsl_multifit_linear_free(work);
//         gsl_matrix_free(X);
//         gsl_vector_free(y);
//         gsl_vector_free(c);
//         gsl_matrix_free(cov);
//         return FAILURE;
//     }
    
//     // 提取回归系数
//     regression_coeffs_.resize(p);
//     for (size_t i = 0; i < p; i++) {
//         regression_coeffs_[i] = gsl_vector_get(c, i);
//     }
    
//     printf("批次 %ld 的拟合优度 (chi^2): %.1f\n", (long)batch_id, chisq);
//     spdlog::info("批次 {} 的拟合优度 (chi^2): {}", batch_id, chisq);
    
//     // 清理GSL资源
//     gsl_multifit_linear_free(work);
//     gsl_matrix_free(X);
//     gsl_vector_free(y);
//     gsl_vector_free(c);
//     gsl_matrix_free(cov);
    
//     // 更新模型状态
//     is_fitted_ = true;
//     return SUCCESS;
// }



// 实现最优多项式回归模型查找函数
// RESPONSE upcite::ConformalRegressor::find_optimal_polynomial_model(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages, 
//     const std::vector<ERROR_TYPE>& errors,
//     ID_TYPE max_degree) {
//     printf("进入find_optimal_polynomial_model，recalls：%zu, coverages：%zu, errors：%zu\n", recalls.size(), coverages.size(), errors.size());

//     // 打印输入的recalls, coverages, errors
//     spdlog::info("find_optimal_polynomial_model输入数据:");
//     for (size_t i = 0; i < recalls.size(); ++i) {
//         spdlog::info("数据点 {}: recall={:.6f}, coverage={:.6f}, error={:.6f}", 
//                     i, recalls[i], coverages[i], errors[i]);
//     }
    
//     // 打印recall,coverage,error到控制台
//     printf("find_optimal_polynomial_model输入数据 (共 %zu 个点):\n", recalls.size());
//     for (size_t i = 0; i < recalls.size(); ++i) {
//         printf("数据点 %zu: recall=%.6f, coverage=%.6f, error=%.6f\n", 
//                i, recalls[i], coverages[i], errors[i]);
//     }
//     if (recalls.size() != coverages.size() || recalls.size() != errors.size() || recalls.empty()) {
//         spdlog::error("输入数据维度不匹配或为空");
//         return FAILURE;
//     }
    
//     const size_t n_samples = recalls.size();
//     double best_mse = std::numeric_limits<double>::max();
//     ID_TYPE best_degree = 0;
    
//     std::vector<double> mse_list;
//     std::vector<double> r2_list;
    
//     // 计算样本数允许的最大多项式阶数
//     // 对于阶数d，特征数量是 1(常数项) + Σ(d+1) 从i=1到d
//     // 即特征数 = 1 + Σ(i+1) 从i=1到d
//     ID_TYPE adjusted_max_degree = max_degree;
    
//     // 计算不同阶数对应的特征数
//     std::vector<size_t> degree_to_features;
//     degree_to_features.push_back(1); // 阶数0只有常数项
    
//     for (ID_TYPE d = 1; d <= max_degree; ++d) {
//         size_t num_features = degree_to_features.back(); // 继承前一阶数的特征数
//         num_features += d + 1; // 当前阶数新增的特征数
//         degree_to_features.push_back(num_features);
//         // 如果特征数超过样本数，就不能使用这个阶数
//         if (num_features >= n_samples) {
//             adjusted_max_degree = d - 1; // 使用前一个阶数
//             break;
//         }
//     }
//     printf("adjusted_max_degree: %ld\n", (long)adjusted_max_degree);
    
//     // 如果所有阶数的特征数都超过样本数，使用最简单的模型（仅常数项）
//     if (adjusted_max_degree < 1) {
//         spdlog::warn("样本数({})过少，无法拟合多项式模型，将使用常数模型", n_samples);
//         printf("警告: 样本数(%zu)过少，无法拟合多项式模型，将使用常数模型\n", n_samples);
        
//         // 使用常数模型（平均值）
//         double mean_error = 0.0;
//         for (const auto& err : errors) {
//             mean_error += err;
//         }
//         mean_error /= errors.size();
        
//         // 设置常数项系数
//         regression_coeffs_.clear();
//         regression_coeffs_.push_back(mean_error);
        
//         printf("使用常数模型，系数: %.6f\n", mean_error);
//         spdlog::info("使用常数模型，系数: {:.6f}", mean_error);
        
//         is_fitted_ = true;
//         return SUCCESS;
//     }
    
//     printf("基于样本数量(%zu)，调整最大多项式阶数: %ld -> %ld\n", 
//            n_samples, (long)max_degree, (long)adjusted_max_degree);
//     spdlog::info("基于样本数量({})，调整最大多项式阶数: {} -> {}", 
//                 n_samples, max_degree, adjusted_max_degree);
    
//     // 测试从1到adjusted_max_degree的多项式
//     for (ID_TYPE degree = 1; degree <= adjusted_max_degree; ++degree) {
//         // 计算当前模型需要的特征数
//         size_t num_features = degree_to_features[degree];
        
//         // 创建设计矩阵
//         gsl_matrix* X = gsl_matrix_alloc(n_samples, num_features);
//         gsl_vector* y = gsl_vector_alloc(n_samples);
//         gsl_vector* c = gsl_vector_alloc(num_features);
//         gsl_matrix* cov = gsl_matrix_alloc(num_features, num_features);
//         double chisq;
        
//         // 填充设计矩阵
//         for (size_t i = 0; i < n_samples; ++i) {
//             // 设置目标变量
//             gsl_vector_set(y, i, errors[i]);
//             // 常数项
//             size_t col_idx = 0;
//             gsl_matrix_set(X, i, col_idx++, 1.0);
//             // 为每个阶构建多项式特征
//             for (ID_TYPE d = 1; d <= degree; ++d) {
//                 for (ID_TYPE p = 0; p <= d; ++p) {
//                     // 计算recall^(d-p) * coverage^p
//                     double feature_val = std::pow(recalls[i], d-p) * std::pow(coverages[i], p);
//                     gsl_matrix_set(X, i, col_idx++, feature_val);
//                 }
//             }
//         }
        
//         // 执行回归计算
//         gsl_multifit_linear_workspace* work = gsl_multifit_linear_alloc(n_samples, num_features);
//         int ret = gsl_multifit_linear(X, y, c, cov, &chisq, work);
        
//         if (ret != GSL_SUCCESS) {
//             spdlog::error("GSL回归计算失败，阶数={}，错误码={}", degree, ret);
//             printf("GSL回归计算失败，阶数=%ld，错误码=%d\n", (long)degree, ret);
//             // 清理资源
//             gsl_multifit_linear_free(work);
//             gsl_matrix_free(X);
//             gsl_vector_free(y);
//             gsl_vector_free(c);
//             gsl_matrix_free(cov);
//             continue;
//         }
        
//         // 计算预测值和误差
//         double mse = 0.0;
//         double ss_total = 0.0;
//         double mean_y = 0.0;
        
//         // 计算y的平均值
//         for (size_t i = 0; i < n_samples; ++i) {
//             mean_y += gsl_vector_get(y, i);
//         }
//         mean_y /= n_samples;
        
//         // 计算预测值和均方误差
//         std::vector<double> predictions(n_samples);
//         for (size_t i = 0; i < n_samples; ++i) {
//             // 手动计算预测值 (X * c)
//             double pred = 0.0;
//             for (size_t j = 0; j < num_features; ++j) {
//                 pred += gsl_matrix_get(X, i, j) * gsl_vector_get(c, j);
//             }
//             predictions[i] = pred;
//             double error = gsl_vector_get(y, i) - pred;
//             mse += error * error;
//             ss_total += std::pow(gsl_vector_get(y, i) - mean_y, 2);
//         }
        
//         mse /= n_samples;
//         double r2 = 1.0 - (mse * n_samples) / ss_total;
        
//         mse_list.push_back(mse);
//         r2_list.push_back(r2);
        
//         printf("多项式阶数 %ld, MSE: %.6f, R^2: %.6f\n", (long)degree, mse, r2);
//         spdlog::info("多项式阶数 {}, MSE: {:.6f}, R^2: {:.6f}", degree, mse, r2);
        
//         // 更新最优模型
//         if (mse < best_mse) {
//             best_mse = mse;
//             best_degree = degree;
            
//             // 保存最优模型系数
//             regression_coeffs_.resize(num_features);
//             for (size_t i = 0; i < num_features; ++i) {
//                 regression_coeffs_[i] = gsl_vector_get(c, i);
//             }
//         }
        
//         // 清理资源
//         gsl_multifit_linear_free(work);
//         gsl_matrix_free(X);
//         gsl_vector_free(y);
//         gsl_vector_free(c);
//         gsl_matrix_free(cov);
//     }
    
//     printf("\n最优多项式阶数: %ld\n", (long)best_degree);
//     printf("最优模型MSE: %.6f\n", best_mse);
//     spdlog::info("最优多项式阶数: {}", best_degree);
//     spdlog::info("最优模型MSE: {:.6f}", best_mse);
    
//     // 输出最优模型系数
//     printf("\n最优模型系数:\n");
//     size_t feature_idx = 0;
//     printf("常数项(β_0): %.6f\n", regression_coeffs_[feature_idx++]);
    
//     // 输出每个阶数的系数，类似Python版本格式
//     for (ID_TYPE d = 1; d <= best_degree; ++d) {
//         for (ID_TYPE p = 0; p <= d; ++p) {
//             if (feature_idx < regression_coeffs_.size()) {
//                 if (d == 1 && p == 0) {
//                     printf("Recall(β_1): %.6f\n", regression_coeffs_[feature_idx++]);
//                 } else if (d == 1 && p == 1) {
//                     printf("Coverage(β_2): %.6f\n", regression_coeffs_[feature_idx++]);
//                 } else {
//                     printf("Recall^%ld·Coverage^%ld: %.6f\n", 
//                            (long)(d-p), (long)p, regression_coeffs_[feature_idx++]);
//                 }
//             }
//         }
//     }
//     is_fitted_ = true;
//     return SUCCESS;
// }




// // 实现多项式模型误差预测方法
// double upcite::ConformalRegressor::predict_polynomial_model_error(
//     double recall, 
//     double coverage, 
//     const std::vector<double>& coeffs,
//     ID_TYPE degree) const {
//     if (coeffs.empty()) {
//         spdlog::error("模型系数为空");
//         return -1.0;
//     }
//     // 计算预测值
//     double predicted_error = coeffs[0];  // 常数项
//     size_t feature_idx = 1;
//     // 计算每个阶数的贡献
//     for (ID_TYPE d = 1; d <= degree; ++d) {
//         for (ID_TYPE p = 0; p <= d; ++p) {
//             if (feature_idx < coeffs.size()) {
//                 // 计算recall^(d-p) * coverage^p
//                 double feature_val = std::pow(recall, d-p) * std::pow(coverage, p);
//                 predicted_error += coeffs[feature_idx++] * feature_val;
//             }
//         }
//     }
//     // 确保预测值非负
//     return std::max(0.0, predicted_error);
// }



// // 训练最优多项式回归模型，用于设置最优误差边界
// RESPONSE upcite::ConformalRegressor::train_optimal_polynomial_model(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages, 
//     const std::vector<ERROR_TYPE>& errors,
//     ID_TYPE max_degree) {
    
//     spdlog::info("进入train_optimal_polynomial_model, recalls: {}, coverages: {}, errors: {}", 
//                 recalls.size(), coverages.size(), errors.size());
//     printf("进入train_optimal_polynomial_model，recalls：%zu, coverages：%zu, errors：%zu\n", 
//           recalls.size(), coverages.size(), errors.size());
    
//     // 检查输入数据是否有效
//     if (recalls.empty() || coverages.empty() || errors.empty()) {
//         spdlog::error("train_optimal_polynomial_model输入数据为空");
//         printf("错误: train_optimal_polynomial_model输入数据为空\n");
//         return FAILURE;
//     }
    
//     // 检查输入数据维度是否匹配
//     if (recalls.size() != coverages.size() || recalls.size() != errors.size()) {
//         spdlog::error("train_optimal_polynomial_model输入数据维度不匹配");
//         printf("错误: train_optimal_polynomial_model输入数据维度不匹配\n");
//         return FAILURE;
//     }
    
//     // 打印输入数据
//     for (size_t i = 0; i < recalls.size(); ++i) {
//         spdlog::info("数据点 {}: recall={:.4f}, coverage={:.4f}, error={:.4f}", 
//                     i, recalls[i], coverages[i], errors[i]);
//     }
    
//     // 调用find_optimal_polynomial_model函数来寻找最优多项式模型
//     RESPONSE result = find_optimal_polynomial_model(recalls, coverages, errors, max_degree);
    
//     if (result != SUCCESS) {
//         spdlog::error("find_optimal_polynomial_model失败");
//         printf("错误: find_optimal_polynomial_model失败\n");
//         return FAILURE;
//     }
    
//     return SUCCESS;
// }


}



// At the end of the file, after the last function
// namespace upcite {

// // Example/testing function for the optimized regional spline methods
// RESPONSE test_optimized_regional_spline() {
//     std::cout << "===== 测试区域优化样条回归 =====" << std::endl;
//     // 创建示例数据
//     std::vector<ERROR_TYPE> recalls = {
//         0.92, 0.95, 0.95, 0.95, 0.99, 1.0, 1.0, 1.0,
//         0.92, 0.93, 0.94, 0.96, 0.97, 0.98, 0.99, 1.0
//     };
    
//     std::vector<ERROR_TYPE> coverages = {
//         1.0, 0.9, 0.95, 0.99, 0.99, 0.95, 0.99, 1.0,
//         0.8, 0.85, 0.9, 0.92, 0.94, 0.96, 0.98, 0.97
//     };
    
//     std::vector<ERROR_TYPE> errors = {
//         0.024, 0.000, 0.188, 0.314, 1.912, 3.563, 6.286, 6.287,
//         0.012, 0.045, 0.123, 0.456, 0.789, 1.234, 3.456, 5.678
//     };
    
//     // 备份原始数据
//     std::vector<ERROR_TYPE> original_recalls = recalls;
//     std::vector<ERROR_TYPE> original_coverages = coverages;
//     std::vector<ERROR_TYPE> original_errors = errors;
    
//     // 创建Conformal回归器
//     ConformalRegressor regressor("spline", 0.95);
    
//     // 拟合优化区域样条模型
//     RESPONSE result = regressor.fit_optimized_regional_spline(
//         recalls, coverages, errors, 0.96, 0.3);
    
//     if (result != SUCCESS) {
//         std::cout << "拟合区域优化样条模型失败" << std::endl;
//         return FAILURE;
//     }
    
//     // 测试预测
//     std::cout << "\n===== 区域优化样条模型预测结果 =====" << std::endl;
//     std::cout << "原始点预测误差对比：" << std::endl;
//     std::cout << "Recall\tCoverage\t实际Error\t预测Error" << std::endl;
    
//     for (size_t i = 0; i < original_recalls.size(); i++) {
//         ERROR_TYPE recall = original_recalls[i];
//         ERROR_TYPE coverage = original_coverages[i];
//         ERROR_TYPE actual_error = original_errors[i];
        
//         // 使用模型预测
//         ERROR_TYPE predicted_error = regressor.predict_regional_spline_error(recall, coverage);
        
//         std::cout << recall << "\t" << coverage << "\t\t" 
//                  << actual_error << "\t\t" << predicted_error << std::endl;
//     }
    
//     // 测试关键点的预测误差
//     std::cout << "\n关键点预测：" << std::endl;
    
//     // 关键点列表
//     std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>> key_points = {
//         {0.95, 0.95}, {0.99, 0.99}, {1.0, 0.95}, {1.0, 0.99}, {1.0, 1.0}
//     };
    
//     for (const auto& point : key_points) {
//         ERROR_TYPE recall = point.first;
//         ERROR_TYPE coverage = point.second;
        
//         // 使用模型预测
//         ERROR_TYPE predicted_error = regressor.predict_regional_spline_error(recall, coverage);
        
//         std::cout << "Recall=" << recall << ", Coverage=" << coverage 
//                  << " => 预测Error: " << predicted_error << std::endl;
//     }
//     return SUCCESS;
//   }
// } // namespace upcite


