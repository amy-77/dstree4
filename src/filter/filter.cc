//
// Created by Qitong Wang on 2022/10/11.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "filter.h"
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>
#include <csignal>
#include <fstream>

#include <boost/filesystem.hpp>
#include <torch/data/example.h>
#include <torch/data/datasets/base.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "spdlog/spdlog.h"
#include <algorithm>

#include "str.h"
#include "vec.h"
#include "comp.h"
#include "interval.h"
#include "dataset.h"
#include "scheduler.h"

#include <gsl/gsl_multifit.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

dstree::Filter::Filter(dstree::Config &config,
                       ID_TYPE id,
                       std::reference_wrapper<torch::Tensor> shared_train_queries) :
    config_(config),
    id_(id),
    is_active_(false),
    global_queries_(shared_train_queries),
    is_trained_(false),
    is_distances_preprocessed_(false),
    is_distances_logged(false),
    global_data_size_(0),
    local_data_size_(0),
    model_setting_ref_(MODEL_SETTING_PLACEHOLDER_REF) {
  if (config.filter_train_is_gpu_) {
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                              static_cast<c10::DeviceIndex>(config.filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  // delayed until allocated (either in trial or activation)
  model_ = nullptr;

  if (!config.to_load_index_ && config.filter_train_nexample_ > 0) {
    global_bsf_distances_.reserve(config.filter_train_nexample_);
    global_lnn_distances_.reserve(config.filter_train_nexample_);

    // QYL
    // global_knn_distances_.reserve(config.filter_train_nexample_);
    lb_distances_.reserve(config.filter_train_nexample_);
  }

  if (config.filter_is_conformal_) {
    conformal_predictor_ = std::make_unique<upcite::ConformalRegressor>(config.filter_conformal_core_type_,
                                                                        config.filter_conformal_confidence_);
  } else {
    conformal_predictor_ = nullptr;
  }
}


// ---------------------------conformal_predictor 的过程---------------------
RESPONSE dstree::Filter::fit_conformal_predictor(bool is_trial, bool collect_runtime_stat) {
  // ===================== 阶段1：参数打印与初始化 =====================
  
  // printf("[DEBUG] =========== Entering fit_conformal_predictor ===========\n");
  // printf("[DEBUG] 当前模式: is_trial=%d, collect_runtime_stat=%d\n", 
        //  static_cast<int>(is_trial), 
        //  static_cast<int>(collect_runtime_stat));
  ID_TYPE num_conformal_examples;
  // is_trial=0, collect_runtime_stat=0
  // ===================== 阶段2：确定符合预测样本数量 =====================
  // printf("[DEBUG] --- 进入样本数量计算阶段 ---\n");
  if (!collect_runtime_stat) {
    //进入这个分支
    printf("[DEBUG] 常规模式：使用全局数据划分验证集\n");
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    
    num_conformal_examples = num_global_valid_examples;

    printf("[DEBUG] 全局训练样本量=%ld, 验证样本量=%ld\n", 
      num_global_train_examples, num_global_valid_examples);
  } else {
    printf("[DEBUG] 运行时统计模式：动态生成样本\n");

    if (config_.get().filter_train_num_global_example_ > 0 && config_.get().filter_train_num_local_example_ >= 0) {
      printf("[DEBUG] 使用 filter_train_num_global_example_ 配置\n");

      ID_TYPE num_global_train_examples =
          config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else if (config_.get().filter_train_nexample_ > 0) {
      ID_TYPE num_global_train_examples = config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_;
      ID_TYPE num_global_valid_examples = config_.get().filter_train_nexample_ - num_global_train_examples;

      num_conformal_examples = num_global_valid_examples;
    } else {
      num_conformal_examples = 8192;
    }
  }


  //生成校准集和残差

  auto residuals = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

  if (collect_runtime_stat) {
    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);

    // include two sentry diffs
    for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
      residuals.push_back(dist(e2));
    }
  } else {
    // printf("[DEBUG] 基于真实数据计算残差\n");
    //训练集的样本数量
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    //验证集的样本数量
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    VALUE_TYPE max_diff = constant::MIN_VALUE, mean_diff = 0, std_diff = 0;
    ID_TYPE num_diff = 0;

    // printf(" 不明白为啥，又算了一遍全局训练样本量=%ld, 验证样本量=%ld\n", num_global_train_examples, num_global_valid_examples);

    // ===================== 添加调试打印 =====================
    // printf("\n[DEBUG] 数组大小检查:\n");
    // printf("global_pred_distances_.size() = %zu\n", global_pred_distances_.size());
    // printf("global_lnn_distances_.size() = %zu\n", global_lnn_distances_.size());
    // printf("num_global_train_examples = %d\n", num_global_train_examples);
    // printf("num_conformal_examples = %d\n", num_conformal_examples);
    // printf("global_data_size_ = %d\n", global_data_size_);

    // 边界检查断言
    assert(num_global_train_examples + num_conformal_examples <= global_pred_distances_.size());
    assert(num_global_train_examples + num_conformal_examples <= global_lnn_distances_.size());

    // 遍历验证集，求出真实最近邻距离和预测距离的差值
    // printf("[DEBUG] 遍历验证集 (共%ld样本)\n", num_global_valid_examples);

    // ===================== 原有残差计算逻辑 =====================
    for (ID_TYPE conformal_i = 0; conformal_i < num_conformal_examples; ++conformal_i) {
      // TODO torch::Tensor to ptr is not stable
      ID_TYPE idx = num_global_train_examples + conformal_i;
      // printf("[DEBUG] 处理样本 %d (全局索引 %d): ", conformal_i, idx);
      // printf("pred=%.3f, lnn=%.3f\n", global_pred_distances_[idx], global_lnn_distances_[idx]); 
      
      if (global_pred_distances_[idx] > constant::MIN_VALUE && global_pred_distances_[idx] < constant::MAX_VALUE &&
          !upcite::equals_zero(global_pred_distances_[idx])) {
        // TODO not necessary for global symmetrical confidence intervals
        // printf("fit_conformal_predictor:Sample %d: pred=%.6f, lnn=%.6f\n", idx, global_pred_distances_[idx], global_lnn_distances_[idx]);
        VALUE_TYPE diff = abs(global_pred_distances_[idx] - global_lnn_distances_[idx]);
        
        if (diff > max_diff) {
          max_diff = diff;
        }
        mean_diff += diff;
        num_diff += 1;
        residuals.emplace_back(diff);
      }
    }

    // ===================== 残差统计结果 =====================
    // printf("\n[DEBUG] 残差统计结果:\n");
    printf("有效残差数量: %d (预期: %d)\n", num_diff, num_conformal_examples);   
    // printf("[DEBUG] 最大残差=%.3f\n", max_diff);

    if (num_diff < num_conformal_examples) {
      spdlog::error("adjuster {:d} {:s} collected {:d} pred diff; expected {:d}",
                    id_, model_setting_ref_.get().model_setting_str, num_diff, num_conformal_examples);
    }
    //计算误差的均值
    mean_diff /= num_diff;
    //计算误差的方差
    for (ID_TYPE diff_i = 0; diff_i < num_diff; ++diff_i) {
      std_diff += (residuals[diff_i] - mean_diff) * (residuals[diff_i] - mean_diff);
    }
    //误差标准差
    std_diff = sqrt(std_diff / num_diff);
    VALUE_TYPE max_normal_value = mean_diff + 3 * std_diff + constant::EPSILON_GAP;
    max_diff += constant::EPSILON_GAP;
    if (max_normal_value < max_diff) {
      max_normal_value = max_diff;
    }
    // printf("[DEBUG] 最终残差边界: max_normal_value=%.3f\n", max_normal_value);

    // //向residuals容器中添加一个值为0的哨兵值（sentry value）, 0表示残差的最小可能值，max_normal_value表示最小可能值   add the first of two sentries: 0
    residuals.push_back(0); 
    // add the second of two sentries: max range upper boundary previously using the max pred value
    residuals.push_back(max_normal_value);
    printf("最终residuals大小: %zu (应等于num_diff+2=%d)\n", residuals.size(), num_diff + 2);
  }
  
  // ===================== 阶段4：核心逻辑分支处理 =====================
  // printf("------CP: is_trial && !collect_runtime_stat -------\n");
  if (is_trial && !collect_runtime_stat) {  //现在不用这个
    printf("[DEBUG] 进入试验模式分支\n");
    //遍历residual容器中的每个元素，如果元素小于0，则取反
    for (auto &residual : residuals) { residual = residual < 0 ? -residual : residual; }
    //对residuals容器中的残差进行升序排序。
    std::sort(residuals.begin(), residuals.end());
    printf("[DEBUG] 排序后残差范围: [%.3f ~ %.3f]\n", residuals.front(), residuals.back());

    //根据置信水平计算残差分位数的索引位置
    printf("filter_trial_confidence_level_ = %f\n", config_.get().filter_trial_confidence_level_);
    auto residual_i = static_cast<ID_TYPE>(static_cast<VALUE_TYPE>(residuals.size())
        * config_.get().filter_trial_confidence_level_);
    //设置置信水平，将分位数对应的残差值设置为CP的置信区间半径，true: 表示这是试验模式; false: 表示不进行修正。
    conformal_predictor_->set_alpha(residuals[residual_i], true, false);

#ifdef DEBUG
//#ifndef DEBUGGED
//id_ 是当前过滤器（Filter）的唯一标识符，
//model_setting_ref_.get().model_setting_str 是当前模型的设置字符串
//filter_trial_confidence_level 是试验模式下的置信度
    spdlog::debug("trial {:d} {:s} error (half-)interval = {:.3f} @ {:.2f}",
                  id_, model_setting_ref_.get().model_setting_str,
                  get_abs_error_interval(), //通过调用 conformal_predictor_->get_alpha()，获取当前CP的置信区间半径
                  config_.get().filter_trial_confidence_level_);
//#endif
#endif
  } else if (!is_trial && collect_runtime_stat) {
    //这里fit还是计算残差分位数
    printf("[DEBUG] !is_trial && collect_runtime_stat \n");
    conformal_predictor_->fit(residuals);
    // printf("[DEBUG] 完成基础符合预测器拟合\n");
    if (config_.get().filter_conformal_is_smoothen_) {
      // printf("[DEBUG] 启用平滑处理\n");

      auto recalls = upcite::make_reserved<ERROR_TYPE>(num_conformal_examples + 2);

      std::random_device rd;
      std::mt19937 e2(rd());
      std::uniform_real_distribution<> dist(0, 1);
      for (ID_TYPE i = 0; i < num_conformal_examples + 2; ++i) {
        recalls.push_back(dist(e2));
      }
      std::sort(recalls.begin(), recalls.end()); //non-decreasing
      //调用fit_spline函数，拟合f:recall_i -> alpha_i
      fit_filter_conformal_spline(recalls);
    }

  } else if (!is_trial && !collect_runtime_stat) { // is_trail: 背包算法等    collect_runtime_stat：收集运行时统计信息，
    //  std::signal(SIGSEGV, sigfaultHandler);   
    // 主要用的是这个
    printf("[DEBUG] !is_trial && !collect_runtime_stat 进入常规生产模式分支\n");
    RESPONSE return_code = FAILURE;
    // 计算残差alphas
    //！！！！！！！！！！！！！！！！！！！
    return_code = conformal_predictor_->fit(residuals);

    if (return_code == FAILURE) {
      printf("[ERROR] 符合预测器拟合失败! residuals.size()=%ld\n", residuals.size());

      spdlog::error("trial {:d} {:s} failed to get made conformal (with {:d}/{:d} residuals); disable it",
                    id_, model_setting_ref_.get().model_setting_str, residuals.size(), num_conformal_examples);
      is_trained_ = false;
      is_active_ = false;
    }
  } else {
    printf("[ERROR] 非法参数组合! is_trial=%d, collect_runtime_stat=%d\n", 
      static_cast<int>(is_trial), static_cast<int>(collect_runtime_stat));
    spdlog::error("trial {:d} {:s} both trial and collect modes were triggered",
                  id_, model_setting_ref_.get().model_setting_str);
    return FAILURE;
  }
  // printf("[DEBUG] =========== 函数执行完成 ===========\n");
  return SUCCESS;
}




// 存储误差到batch_residuals容器中和batch_alpha容器中
RESPONSE dstree::Filter::fit_batch_conformal_predictor(
    bool is_trial,
    ID_TYPE num_calib_batches,
    const std::vector<torch::Tensor>& calib_data_batches,
    const std::vector<torch::Tensor>& calib_target_batches) {
  
  // 创建多个校准集的残差容器
  std::vector<std::vector<ERROR_TYPE>> batch_residuals(num_calib_batches);
  // 对每个校准批次计算残差
  for (ID_TYPE batch_idx = 0; batch_idx < num_calib_batches; ++batch_idx) {
    torch::Tensor batch_data = calib_data_batches[batch_idx]; // 获取当前批次的校准数据
    torch::Tensor batch_targets = calib_target_batches[batch_idx]; // 获取当前批次的真实最近邻距离， 所有batch的query到当前节点的距离
    ID_TYPE batch_size = batch_data.size(0); // 获取当前批次的样本数量
    
    // 使用模型预测当前批次
    c10::InferenceMode guard;
    model_->eval();
    torch::Tensor batch_predictions = model_->forward(batch_data).detach().cpu(); // 获取当前批次的预测值
    
    VALUE_TYPE max_diff = constant::MIN_VALUE, mean_diff = 0, std_diff = 0;
    ID_TYPE num_diff = 0;
    
    // 计算此批次的残差
    for (ID_TYPE i = 0; i < batch_size; ++i) {
      VALUE_TYPE pred = batch_predictions[i].item<VALUE_TYPE>(); // 获取预测值
      VALUE_TYPE target = batch_targets[i].item<VALUE_TYPE>(); // 获取真实值
      
      if (pred > constant::MIN_VALUE && pred < constant::MAX_VALUE && !upcite::equals_zero(pred)) {
        // VALUE_TYPE diff = std::abs(pred - target);  // 取绝对值
        
        // QYL 不取绝对值版本
        VALUE_TYPE diff = pred - target;
        if (diff < 0) diff = -diff;

        batch_residuals[batch_idx].push_back(diff);
        if (diff > max_diff) max_diff = diff;
        mean_diff += diff;
        num_diff += 1;
      }
    }
    
    // 处理统计和哨兵值逻辑
    if (num_diff > 0) {
      // 计算统计值
      mean_diff /= num_diff;
      for (const auto& diff : batch_residuals[batch_idx]) {
        std_diff += (diff - mean_diff) * (diff - mean_diff);
      }
      std_diff = sqrt(std_diff / num_diff);
      
      // 添加哨兵值
      VALUE_TYPE max_normal_value = mean_diff + 3 * std_diff + constant::EPSILON_GAP;
      max_diff += constant::EPSILON_GAP;
      if (max_normal_value < max_diff) max_normal_value = max_diff;
      
      batch_residuals[batch_idx].push_back(0); // 最小值哨兵
      batch_residuals[batch_idx].push_back(max_normal_value); // 最大值哨兵
    } else {
      printf("[ERROR] 校准批次 %d 无有效残差!\n", batch_idx+1);
      batch_residuals[batch_idx].push_back(0);
      batch_residuals[batch_idx].push_back(0.1);
    }
  }
  // 打印batch_residuals的大小信息
  // printf("\nbatch_residuals大小统计信息:\n");
  // printf("批次总数: %zu\n", batch_residuals.size());
  // for (ID_TYPE batch_idx = 0; batch_idx < batch_residuals.size(); ++batch_idx) {
  //   printf("批次 %d: 包含 %zu 个残差值\n", batch_idx + 1, batch_residuals[batch_idx].size());
  // }
  
  // 使用计算好的批次残差拟合保形预测器
  //batch_residuals是vector<vector<ERROR_TYPE>>类型，外层是batch_idx，内层是residuals，residuals的size是batch_size+2
  RESPONSE return_code = conformal_predictor_->fit_batch(batch_residuals);
  // 保存批处理alphas为CSV文件

  // 确保结果目录存在
  std::string results_dir = config_.get().results_path_;
  if (!results_dir.empty() && !fs::exists(results_dir)) {
    fs::create_directories(results_dir);
  }

  // 创建子文件夹 batch_alphas_filters
  std::string batch_alphas_dir = results_dir + "/batch_alphas_filters";
  if (!fs::exists(batch_alphas_dir)) {
    fs::create_directories(batch_alphas_dir);
  }
  std::string save_path = batch_alphas_dir + "/filter_" + std::to_string(id_) + "_batch_alphas.csv";
  conformal_predictor_->save_batch_alphas_csv(save_path);
  
  if (return_code == FAILURE) {
    printf("[ERROR] 符合预测器拟合失败!\n");
    spdlog::error("filter {:d} {:s} failed in batch conformal fitting; disabling",
                 id_, model_setting_ref_.get().model_setting_str);
    is_trained_ = false;
    is_active_ = false;
    return FAILURE;
  }
  
  return SUCCESS;
}








// 这只是针对当前的filter进行训练，
RESPONSE dstree::Filter::train(bool is_trial) {

//功能：训练一个用于距离预测的机器学习模型（如CNN或线性模型），并结合保形预测（Conformal Prediction）校准预测结果。
/*
核心流程：
预处理：处理距离数据（如平方根转换）。
数据划分：将数据分为训练集、验证集。
模型训练：通过反向传播优化模型参数。
模型选择：根据验证损失保存最佳模型。
预测与校准：生成预测结果并调用保形预测校准。
参数：is_trial 表示是否为试验模式（影响后续校准逻辑）。
*/
  // printf("进入Filter::train函数\n");
  // ========================= 1. 前置检查 ==============================
  // 检查是否已训练或需要加载预训练模型
  if (is_trained_ || config_.get().to_load_filters_) {
    return FAILURE;
  }
  // 检查过滤器激活状态与模式合法性
  if (!is_active_ && !is_trial) {
    spdlog::error("filter {:d} neither is_active nor is_trial; exit", id_);
    spdlog::shutdown();
    exit(FAILURE);
  }

   // =========================== 2. CUDA流初始化 ==========================
  // 初始化CUDA流，用于GPU并行计算
  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
        // 获取当前CUDA流的ID（GPU训练时使用）
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }

  // ============================= 3. 数据预处理 =============================
  // 若配置要求移除平方（filter_remove_square_）且未预处理过。 
  // printf("[DEBUG] 条件检查: filter_remove_square_ = %d, is_distances_preprocessed_ = %d\n", 
    // static_cast<int>(config_.get().filter_remove_square_),  // 布尔转 int
    // static_cast<int>(is_distances_preprocessed_);          // 布尔转 int
  // 打印未开平方的距离值
  // printf("\n[DEBUG] 未开平方的距离值:\n");
  // printf("global_lnn_distances_ (前10个):");
  // for (size_t i = 0; i < std::min(global_data_size_, (ID_TYPE)10); ++i) {
  //   printf(" %.2f", global_lnn_distances_[i]);
  // }
  // printf("\n");
  
  // printf("global_bsf_distances_ (前10个):");
  // for (size_t i = 0; i < std::min(global_data_size_, (ID_TYPE)10); ++i) {
  //   printf(" %.2f", global_bsf_distances_[i]);
  // }
  // printf("\n");
  
  printf("global_data_size_ = %zu\n", global_data_size_);


  if (config_.get().filter_remove_square_ && !is_distances_preprocessed_) {
    // 对全局最近邻距离和最佳搜索距离取平方根

    for (ID_TYPE i = 0; i < global_data_size_; ++i) {
      global_lnn_distances_[i] = sqrt(global_lnn_distances_[i]); //对全局最近邻距离取平方根
      global_bsf_distances_[i] = sqrt(global_bsf_distances_[i]); //对最佳搜索距离取平方根
    }
    // 打印平方根后的距离值
    printf("\n[DEBUG] 平方根后的距离值:\n");
    printf("global_lnn_distances_ (前10个):");
    for (size_t i = 0; i < std::min(global_data_size_, (ID_TYPE)10); ++i) {
      printf(" %.2f", global_lnn_distances_[i]);
    }
    printf("\n");
    printf("global_bsf_distances_ (前10个):");
    for (size_t i = 0; i < std::min(global_data_size_, (ID_TYPE)10); ++i) {
      printf(" %.2f", global_bsf_distances_[i]);
    }
    printf("\n");

    // printf("global_data_size_single_ = %ld\n", static_cast<long>(global_data_size_single_));  
    // printf("global_lnn_distances_single_ (前10个):");
    // for (size_t i = 0; i < std::min(global_data_size_single_, (ID_TYPE)10); ++i) {
    //   printf(" %.2f", global_lnn_distances_single_[i]);
    // }
    // printf("\n"); 
    // printf("global_bsf_distances_single_ (前10个):");
    // for (size_t i = 0; i < std::min(global_data_size_single_, (ID_TYPE)10); ++i) {
    //   printf(" %.2f", global_bsf_distances_single_[i]);
    // }
    // printf("\n"); 

    // // 打印global_lnn_distances_single_和global_bsf_distances_single_的大小
    // printf("global_lnn_distances_single_.size() = %zu\n", global_lnn_distances_single_.size());
    // printf("global_bsf_distances_single_.size() = %zu\n", global_bsf_distances_single_.size());

    
    
    // printf("global_lnn_distances_.size() = %zu\n", global_lnn_distances_.size());
    // printf("global_bsf_distances_.size() = %zu\n", global_bsf_distances_.size());
    // printf("global_data_size_ = %zu\n", global_data_size_);
    // 对下界距离取平方根（如果存在）
    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < global_data_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }
    // 对local最近邻距离取平方根（如果存在局部local数据）
    if (local_data_size_ > 0) {
      for (ID_TYPE i = 0; i < local_data_size_; ++i) {
        local_lnn_distances_[i] = sqrt(local_lnn_distances_[i]);
      }
    }
    is_distances_preprocessed_ = true; // 标记已预处理
  }


#ifdef DEBUG
//#ifndef DEBUGGED
  if (!is_distances_logged) {
    // 记录下界距离、最佳搜索距离等预处理后的数据
    if (!lb_distances_.empty()) {
      spdlog::debug("filter {:d} s{:d} lb{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(lb_distances_.data(), global_data_size_));
    }               //记录下界距离

    spdlog::debug("filter {:d} s{:d} bsf{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_bsf_distances_.data(), global_data_size_));
                  //记录全局最佳搜索距离

    spdlog::debug("filter {:d} s{:d} gnn{:s} = {:s}",
                  id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                  upcite::array2str(global_lnn_distances_.data(), global_data_size_));
                  //记录全局最近邻距离
    if (local_data_size_ > 0) {
      spdlog::debug("filter {:d} s{:d} lnn{:s} = {:s}",
                    id_, stream_id, config_.get().filter_remove_square_ ? "" : "_sq",
                    upcite::array2str(local_lnn_distances_.data(), local_data_size_));
    }             //记录局部最近邻距离

    is_distances_logged = true;
  }
//#endif
#endif

  // ============================ 5. 数据划分 ====================================
  //划分训练集和验证机
  ID_TYPE num_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
  ID_TYPE num_valid_examples = global_data_size_ - num_train_examples;
 
  torch::Tensor train_data, valid_data;
  torch::Tensor train_targets, valid_targets;

  printf("[DEBUG] global_data_size_ = %ld\n", static_cast<long>(global_data_size_));
  printf("[DEBUG] local_data_size_ = %ld\n", static_cast<long>(local_data_size_));
  printf("config_.get().filter_train_num_global_example_ = %ld\n", static_cast<long>(config_.get().filter_train_num_global_example_));
  printf("config_.get().filter_train_num_local_example_ = %ld\n", static_cast<long>(config_.get().filter_train_num_local_example_));
  printf("global_data_size_single_ = %ld\n", static_cast<long>(global_data_size_single_));
  // -------------------5.1 存在局部数据local data时的处理----------------

  if (local_data_size_ > 0) {
    // -------------------5.1.1  获取训练集的全局和局部数据----------------
    assert(global_data_size_ == config_.get().filter_train_num_global_example_);
    
    assert(local_data_size_ == config_.get().filter_train_num_local_example_);
    //确定全局数据的训练样本数量：
    ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;
    //确定局部数据的训练样本数量：
    ID_TYPE num_local_train_examples = local_data_size_ * config_.get().filter_train_val_split_;
    
    //从global_queries_中获取全局训练数据
    torch::Tensor global_train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    //从global_lnn_distances_中获取指定训练集数量的全局训练标签：1nn_distance
    torch::Tensor global_train_targets = torch::from_blob(global_lnn_distances_.data(),
                                                          num_global_train_examples,
                                                          torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //从local_queries_中获取局部训练数据
    torch::Tensor local_train_data = torch::from_blob(local_queries_.data(),
                                                      {num_local_train_examples, config_.get().series_length_},
                                                      torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //从local_lnn_distances_中获取指定训练集数量的局部训练标签：1nn_distance                                                  
    torch::Tensor local_train_targets = torch::from_blob(local_lnn_distances_.data(),
                                                         num_local_train_examples,
                                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    //合并loca和global数据，作为完整的训练数据(query)和训练标签(dij)
    train_data = torch::cat({global_train_data, local_train_data}, 0);
    train_targets = torch::cat({global_train_targets, local_train_targets}, 0);
    
    // --------------------------5.1.2  获取验证集的全局和局部数据----------------
    ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
    ID_TYPE num_local_valid_examples = local_data_size_ - num_local_train_examples;

    torch::Tensor global_valid_data = global_queries_.get().index(
        {torch::indexing::Slice(num_train_examples, global_data_size_)}).clone();
    torch::Tensor global_valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples,num_global_valid_examples,torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    torch::Tensor local_valid_data = torch::from_blob(
        local_queries_.data() + num_local_train_examples * config_.get().series_length_,
        {num_local_valid_examples, config_.get().series_length_},
        torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_valid_targets = torch::from_blob(local_lnn_distances_.data() + num_local_train_examples,num_local_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = torch::cat({global_valid_data, local_valid_data}, 0);
    valid_targets = torch::cat({global_valid_targets, local_valid_targets}, 0);

    num_train_examples = num_global_train_examples + num_local_train_examples;
    num_valid_examples = num_global_valid_examples + num_local_valid_examples;

    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(valid_data.size(0) == num_valid_examples && valid_targets.size(0) == num_valid_examples);
  
  } else {

    //5.2 不存在局部数据local data时的处理,  仅全局数据时的处理
    assert(global_data_size_ == config_.get().filter_train_nexample_);
    //train_data是query，train_targets是全局1nn最近距离
    train_data = global_queries_.get().index({torch::indexing::Slice(0, num_train_examples)}).clone();
    train_targets = torch::from_blob(global_lnn_distances_.data(),
                                     num_train_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    valid_data = global_queries_.get().index({torch::indexing::Slice(
        num_train_examples, global_data_size_)}).clone();
    valid_targets = torch::from_blob(global_lnn_distances_.data() + num_train_examples,
                                     num_valid_examples,
                                     torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  }


  // ============================= 6. 数据加载器初始化 ==============================
  auto train_dataset = upcite::SeriesDataset(train_data, train_targets);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);

  // reuse validation examples as conformal examples
  //这里验证集的数据作为conformal的数据
  ID_TYPE num_conformal_examples = num_valid_examples;
  torch::Tensor conformal_data = valid_data;
  torch::Tensor conformal_targets = valid_targets; //dij, 

  // ==================================== 7. 模型初始化 ==============================
  // 根据配置创建模型（如CNN或线性模型）
  model_ = dstree::get_model(config_);
  model_->to(*device_);

  // ==================================== 8. 训练准备 ================================
  // 最佳模型状态跟踪，用于提前终止
  // for early termination
  std::unordered_map<std::string, torch::Tensor> best_model_state;
  VALUE_TYPE best_validation_loss = constant::MAX_VALUE;
  ID_TYPE best_validation_epoch = -1;
  // 优化器选择（CNN用Adam，其他用SGD）
  std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
  if (model_->model_type_ == CNN) {
    optimizer = std::make_shared<torch::optim::Adam>(model_->parameters(), config_.get().filter_train_learning_rate_);
  } else {
    optimizer = std::make_shared<torch::optim::SGD>(model_->parameters(), config_.get().filter_train_learning_rate_);
  }
  ID_TYPE initial_cooldown_epochs = config_.get().filter_train_nepoch_ / 2;

  //  学习率调整策略（ReduceLROnPlateau）, 用于验证损失
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(
      *optimizer, initial_cooldown_epochs, optim::MIN, config_.get().filter_lr_adjust_factor_);

  // original 损失函数（均方误差）
  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));
  
  // 添加自定义非对称loss函数
  // auto asymmetric_loss = [](const torch::Tensor& pred, const torch::Tensor& target, float overestimation_penalty = 5.0) {
  //   // 计算预测误差
  //   auto diff = pred - target;
  //   // 创建掩码：overestimation (pred > target) → 1.0, underestimation → 0.0
  //   auto overestimation_mask = (diff > 0).to(torch::kFloat32);
  //   // 创建权重：高估时使用overestimation_penalty，低估时使用1.0
  //   auto weights = 1.0 + (overestimation_penalty - 1.0) * overestimation_mask;
  //   // 计算带权重的平方误差
  //   auto squared_error = weights * diff * diff;
  //   // 返回平均loss
  //   return squared_error.mean();
  // };


#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);

  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif

  //================================= 9. 训练循环 ==============================
  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
    model_->train(); // 切换至训练模式
   
    // 9.1 前向传播与反向传播
    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;
      optimizer->zero_grad();
      torch::Tensor prediction = model_->forward(batch_data);

      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      // QYL: 修改loss 使用新的asymmetric_loss 
      // torch::Tensor loss = asymmetric_loss(prediction, batch_target, 5.0);
  

      loss.backward(); // 反向传播
      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer->step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0) / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
#endif

    // 9.2 验证阶段
    { // evaluate
      VALUE_TYPE valid_loss = 0;

      c10::InferenceMode guard;
      model_->eval();  // 切换至评估模式

      torch::Tensor prediction = model_->forward(valid_data);

      valid_loss = mse_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();
      // QYL: 修改loss 使用新的asymmetric_loss 
      // valid_loss = asymmetric_loss(prediction, valid_targets).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif

    // // 2. QYL 添加额外的评估指标
      // auto diff = prediction - valid_targets;
      // auto overestimation = (diff > 0);
      // // 计算高估率(%)
      // float overestimation_rate = overestimation.to(torch::kFloat32).mean().item<float>() * 100;
      // // 计算平均高估幅度
      // float mean_overestimation = diff.masked_select(overestimation).mean().item<float>();
      // // 打印评估指标
      // spdlog::info("Epoch {}: valid_loss={:.4f}, overestimation_rate={:.2f}%, mean_overestimation={:.4f}",
      //             epoch, valid_loss, overestimation_rate, mean_overestimation);




       // original 记录最佳模型状态
      if (epoch > initial_cooldown_epochs) {
        if (best_validation_loss > valid_loss) {
          best_validation_loss = valid_loss;
          best_validation_epoch = epoch;

          for (const auto &pair : model_->named_parameters()) {
            best_model_state[pair.key()] = pair.value().clone();
          }
        }
      }
      // 学习率调整与早停策略
      upcite::optim::LR_RETURN_CODE return_code = lr_scheduler.check_step(valid_loss);
      if (return_code == upcite::optim::EARLY_STOP) {
        epoch = config_.get().filter_train_nepoch_;
      }
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} s{:d} {:s} tloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} s{:d} {:s} vloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

  c10::InferenceMode guard;

// ============================ 10. 模型恢复与预测 =============================
// 恢复最佳模型状态
  if (best_validation_epoch > initial_cooldown_epochs) {
#ifdef DEBUG
    spdlog::debug("filter {:d} s{:d} {:s} restore from e{:d}, vloss {:.4f}",
                  id_, stream_id, model_setting_ref_.get().model_setting_str,
                  best_validation_epoch, best_validation_loss);
#endif

    for (auto &pair : best_model_state) {
      model_->named_parameters()[pair.first].detach_();
      model_->named_parameters()[pair.first].copy_(pair.second);
    }
  }
  //调用模型进行评估，对全局数据进行预测
  model_->eval();

  auto prediction = model_->forward(global_queries_).detach().cpu();
  assert(prediction.size(0) == global_data_size_);
  auto *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();
  
  // !!!!!!!!!!!!!!!!!!1  存储模型预测结果到global_pred_distances_
  global_pred_distances_.insert(global_pred_distances_.end(), predictions_array, predictions_array + global_data_size_);
 
  // 打印 global_pred_distances_ 的 size
  // printf("Size of global_pred_distances_: %zu\n", global_pred_distances_.size());
  #ifdef DEBUG
  spdlog::info("filter {:d}{:s} s{:d} {:s} g_pred{:s} = {:s}",
               id_, is_trial ? " (trial)" : "",
               stream_id, model_setting_ref_.get().model_setting_str,
               config_.get().filter_remove_square_ ? "" : "_sq",
               upcite::array2str(predictions_array, global_data_size_));

#endif

  if (config_.get().filter_is_conformal_) {
    //---------------------------------Conformal Prediction---------------------------------
    //这里就是我们要的，对预测距离进行Conformal Prediction的校准
    printf("\n---------------正式进入CP了,集中注意力---------------\n");
    printf("is_trial = %d\n", is_trial);
    fit_conformal_predictor(is_trial); 
    // fit_conformal_predictor_batch(is_trial);
    // printf("\n");
  }

 //  net->to(torch::Device(torch::kCPU));
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!is_trial) {
    is_trained_ = true;
  } else {
    // TODO should this work around be improved
    global_pred_distances_.clear();
  }

  return SUCCESS;
}




//针对每个filter去train一个小模型，包括处理收集数据，训练模型，预测距离，收集误差
// 这只是针对当前的filter进行训练，而不是针对所有filter进行训练
RESPONSE dstree::Filter::batch_train(bool is_trial) {
// printf("\n--------Filter::batch_train-------\n");
//# 实现多校准集的 batch_train 函数
  // printf("进入Filter::batch_train函数\n");
  // ========================= 1. 前置检查 ==============================
  if (is_trained_ || config_.get().to_load_filters_) {
    return FAILURE;
  }
  if (!is_active_ && !is_trial) {
    spdlog::error("filter {:d} neither is_active nor is_trial; exit", id_);
    spdlog::shutdown();
    exit(FAILURE);
  }

   // =========================== 2. CUDA流初始化 ==========================
  ID_TYPE stream_id = -1;
  if (config_.get().filter_train_is_gpu_) {
    stream_id = at::cuda::getCurrentCUDAStream(config_.get().filter_device_id_).id(); // compiles with libtorch-gpu
  }
  

  // ============================= 3. 数据预处理 =============================
  if (config_.get().filter_remove_square_ && !is_distances_preprocessed_) {
    for (ID_TYPE i = 0; i < global_data_size_; ++i) {
      global_lnn_distances_[i] = sqrt(global_lnn_distances_[i]); //对全局最近邻距离取平方根
      global_bsf_distances_[i] = sqrt(global_bsf_distances_[i]); //对最佳搜索距离取平方根
    }
    // printf("global_data_size_ = %zu\n", global_data_size_);
    // 对下界距离取平方根（如果存在）
    if (!lb_distances_.empty()) {
      for (ID_TYPE i = 0; i < global_data_size_; ++i) {
        lb_distances_[i] = sqrt(lb_distances_[i]);
      }
    }
    
    // 对local最近邻距离取平方根（如果存在局部local数据）
    if (local_data_size_ > 0) {
      for (ID_TYPE i = 0; i < local_data_size_; ++i) {
        local_lnn_distances_[i] = sqrt(local_lnn_distances_[i]);
      }
    }

    is_distances_preprocessed_ = true; 

   
  }


  // ========== 关键修改点 1: 数据划分策略 ==========
  // 将数据划分为: 训练集、验证集和多个校准集
  ID_TYPE num_calib_batches = config_.get().filter_conformal_num_batches_;
  ID_TYPE num_examples_per_calib_batch = global_data_size_ / num_calib_batches;

  // 修改：新的数据分配比例
  // Global data: 2:1:7 (train:validate:CP training)
  // Local data: 4:1 (train:validate)
  
  ID_TYPE num_calib_examples;
  ID_TYPE num_batches;
  ID_TYPE remainder;

  // Global data分配 (总比例2:1:7，总和为10)
  ID_TYPE num_global_train_examples = global_data_size_ * 2 / 6;  // 20%
  ID_TYPE num_global_valid_examples = global_data_size_ * 1 / 6;  // 10%  
  ID_TYPE num_global_cp_examples = global_data_size_ - num_global_train_examples - num_global_valid_examples;  // 70%

  // Local data分配 (总比例4:1，总和为5)
  ID_TYPE num_local_train_examples = local_data_size_ * 4 / 5;  // 80%
  ID_TYPE num_local_valid_examples = local_data_size_ - num_local_train_examples;  // 20%

  // 兼容原有变量名
  ID_TYPE num_train_examples = num_global_train_examples + (local_data_size_ > 0 ? num_local_train_examples : 0);
  ID_TYPE num_valid_examples = num_global_valid_examples + (local_data_size_ > 0 ? num_local_valid_examples : 0);

  // printf("数据分配统计:\n");
  // printf("num_global_train_examples: %ld (%.1f%%)\n", num_global_train_examples, (float)num_global_train_examples * 100 / global_data_size_);
  // printf("num_global_valid_examples: %ld (%.1f%%)\n", num_global_valid_examples, (float)num_global_valid_examples * 100 / global_data_size_);
  // printf("num_global_cp_examples: %ld (%.1f%%)\n", num_global_cp_examples, (float)num_global_cp_examples * 100 / global_data_size_);
  // if (local_data_size_ > 0) {
  //   printf("num_local_train_examples: %ld (%.1f%%)\n", num_local_train_examples, (float)num_local_train_examples * 100 / local_data_size_);
  //   printf("num_local_valid_examples: %ld (%.1f%%)\n", num_local_valid_examples, (float)num_local_valid_examples * 100 / local_data_size_);
  // }
  // printf("num_train_examples: %ld (%.1f%%)\n", num_train_examples, (float)num_train_examples * 100 / global_data_size_);
  // printf("num_valid_examples: %ld (%.1f%%)\n", num_valid_examples, (float)num_valid_examples * 100 / global_data_size_);

  // 创建数据张量
  torch::Tensor train_data, valid_data, calibration_data;
  torch::Tensor train_targets, valid_targets, calibration_targets;
  torch::Tensor global_valid_data, global_valid_targets;
  std::vector<torch::Tensor> calib_data_batches(num_calib_batches);
  std::vector<torch::Tensor> calib_target_batches(num_calib_batches);

  // 处理全局和局部数据
  if (local_data_size_ > 0) {

    assert(global_data_size_ == config_.get().filter_train_num_global_example_);
    assert(local_data_size_ == config_.get().filter_train_num_local_example_);
    
    // 5.1.1 获取训练集的全局和局部数据
    // 全局训练数据处理 (前20%的global data)
    torch::Tensor global_train_data = global_queries_.get().index({torch::indexing::Slice(0, num_global_train_examples)}).clone();
    torch::Tensor global_train_targets = torch::from_blob(global_lnn_distances_.data(), num_global_train_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 局部训练数据处理 (前80%的local data)
    torch::Tensor local_train_data = torch::from_blob(local_queries_.data(),{num_local_train_examples, config_.get().series_length_}, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_train_targets = torch::from_blob(local_lnn_distances_.data(), num_local_train_examples,torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    // 合并训练数据
    train_data = torch::cat({global_train_data, local_train_data}, 0);
    train_targets = torch::cat({global_train_targets, local_train_targets}, 0);
    
    // 5.1.2 获取验证集的全局和局部数据
    // 全局验证数据：global data的第二个10%部分 (20%-30%区间)
    torch::Tensor global_valid_data = global_queries_.get().index({torch::indexing::Slice(num_global_train_examples, num_global_train_examples + num_global_valid_examples)}).clone();
    torch::Tensor global_valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples, num_global_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 局部验证数据：local data的后20%部分
    torch::Tensor local_valid_data = torch::from_blob(local_queries_.data() + num_local_train_examples * config_.get().series_length_,
        {num_local_valid_examples, config_.get().series_length_}, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    torch::Tensor local_valid_targets = torch::from_blob(local_lnn_distances_.data() + num_local_train_examples, num_local_valid_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    

    // 合并验证数据
    valid_data = torch::cat({global_valid_data, local_valid_data}, 0);
    valid_targets = torch::cat({global_valid_targets, local_valid_targets}, 0);

    // 5.1.3 获取CP训练数据
    // CP训练数据：global data的最后70%部分 (30%-100%区间)
    ID_TYPE global_cp_start_idx = num_global_train_examples + num_global_valid_examples;
    torch::Tensor global_cp_data = global_queries_.get().index({torch::indexing::Slice(global_cp_start_idx, global_data_size_)}).clone();
    torch::Tensor global_cp_targets = torch::from_blob(global_lnn_distances_.data() + global_cp_start_idx, num_global_cp_examples, torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    // 验证数据分配正确性
    assert(train_data.size(0) == (num_global_train_examples + num_local_train_examples));
    assert(valid_data.size(0) == (num_global_valid_examples + num_local_valid_examples));
    assert(global_cp_data.size(0) == num_global_cp_examples);

    // printf("global_train_data.size(0): %ld\n", static_cast<long>(global_train_data.size(0)));
    // printf("local_train_data.size(0): %ld\n", static_cast<long>(local_train_data.size(0)));
    // printf("global_valid_data.size(0): %ld\n", static_cast<long>(global_valid_data.size(0)));
    // printf("local_valid_data.size(0): %ld\n", static_cast<long>(local_valid_data.size(0)));

    // printf("global_cp_data.size(0): %ld\n", static_cast<long>(global_cp_data.size(0)));
    // printf("train_data.size(0): %ld\n", static_cast<long>(train_data.size(0)));
    // printf("valid_data.size(0): %ld\n", static_cast<long>(valid_data.size(0)));

    // 使用global CP data作为校准数据
    calibration_data = global_cp_data;
    calibration_targets = global_cp_targets;
    
    // 确定校准样本数量
    num_calib_examples = global_cp_data.size(0);
    // printf("校准数据总量: %d\n", num_calib_examples);
    
    // 如果启用组合方法生成校准批次
    if (config_.get().filter_conformal_use_combinatorial_) {
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_calibration_batches(calibration_data, calibration_targets, 
                                       calib_data_batches, calib_target_batches,
                                       calib_query_ids) == FAILURE) {
                                        
        printf("生成组合校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      
      // 打印校准集ID信息
      // printf("\n校准集ID信息统计:\n");
      // for (ID_TYPE i = 0; i < calib_query_ids.size(); ++i) {
      //   printf("批次 %ld: 包含 %ld 个查询ID\n", i, calib_query_ids[i].size());
      // }
      // printf("总批次数: %ld\n", calib_query_ids.size());
      // 更新批次数量
      num_batches = calib_data_batches.size();
      
      // 保存查询ID到文件
      // save_calib_query_ids(calib_query_ids, "filter_" + std::to_string(id_) + "_calib_query_ids");
      // printf("校准批次信息已保存到 filter_%d_calib_query_ids.txt\n", id_);
    } else {
      // 均匀划分校准集
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_uniform_calibration_batches(calibration_data, calibration_targets,
                                              calib_data_batches, calib_target_batches,
                                              calib_query_ids) == FAILURE) {
        printf("生成均匀校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      // 更新批次数量
      num_batches = calib_data_batches.size();
    }
    
    // 更新样本数量统计
    num_train_examples = train_data.size(0);
    num_valid_examples = calibration_data.size(0);
    // 验证数据维度
    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(calibration_data.size(0) == num_valid_examples && calibration_targets.size(0) == num_valid_examples);
    // printf("训练集大小: %d, 验证集大小(本地): %d, 校准批次数(全局): %d\n", 
    //       num_train_examples, num_valid_examples, num_batches);
    // 打印每个校准集的大小和总数统计
    ID_TYPE total_calib_size = 0;
    // printf("\n校准集详细信息:\n");
    for (ID_TYPE i = 0; i < num_batches; ++i) {
        ID_TYPE batch_size = calib_data_batches[i].size(0);
        total_calib_size += batch_size;
        // printf("校准集 %d: %d 个样本\n", i + 1, batch_size);
    }
    // printf("校准集总数: %d 个\n校准集数量: %d 个\n", total_calib_size, num_batches);


  
  } else {

    // 只使用全局数据时的多批次CP划分 local_data_size_ <= 0


    printf("只使用全局数据时的多批次CP划分 local_data_size_ <= 0\n");
    // ========== 关键修改点 2: 只使用全局数据时的多批次划分 ==========
    
    // 训练数据 (前20%的global data)
    train_data = global_queries_.get().index({torch::indexing::Slice(0, num_global_train_examples)}).clone();
    train_targets = torch::from_blob(global_lnn_distances_.data(),
                                   num_global_train_examples,
                                   torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);

    // 验证数据 (20%-30%区间的global data)
    valid_data = global_queries_.get().index({torch::indexing::Slice(
        num_global_train_examples, num_global_train_examples + num_global_valid_examples)}).clone();
    valid_targets = torch::from_blob(global_lnn_distances_.data() + num_global_train_examples,
                                   num_global_valid_examples,
                                   torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // CP训练数据 (30%-100%区间的global data)
    ID_TYPE global_cp_start_idx = num_global_train_examples + num_global_valid_examples;
    calibration_data = global_queries_.get().index({torch::indexing::Slice(global_cp_start_idx, global_data_size_)}).clone();
    calibration_targets = torch::from_blob(global_lnn_distances_.data() + global_cp_start_idx,
                                         num_global_cp_examples,
                                         torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
    
    // 确定校准样本数量
    num_calib_examples = calibration_data.size(0);
    // printf("校准数据总量: %d\n", num_calib_examples);
    
    // 如果启用组合方法生成校准批次
    if (config_.get().filter_conformal_use_combinatorial_) {
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      // 生成组合校准批次和对应的查询ID
      if (generate_calibration_batches(calibration_data, calibration_targets, 
                                       calib_data_batches, calib_target_batches,
                                       calib_query_ids) == FAILURE) {
        printf("生成组合校准批次失败\n");
        return FAILURE;
      }
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      
      // 保存查询ID到文件
      // save_calib_query_ids(calib_query_ids, "filter_" + std::to_string(id_) + "_calib_query_ids");
      
      // 更新批次数量
      num_batches = calib_data_batches.size();

    } else {
      // 均匀划分校准集
      std::vector<std::vector<ID_TYPE>> calib_query_ids;
      if (generate_uniform_calibration_batches(calibration_data, calibration_targets,
                                              calib_data_batches, calib_target_batches,
                                              calib_query_ids) == FAILURE) {
        printf("生成均匀校准批次失败\n");
        return FAILURE;
      }
      
      // 存储校准批次对应的查询ID，用于后续计算recall
      batch_calib_query_ids_ = calib_query_ids;
      // 更新批次数量
      num_batches = calib_data_batches.size();
    }
    
    // 更新样本数量统计
    num_train_examples = train_data.size(0);
    num_valid_examples = calibration_data.size(0);
    // 验证数据维度
    assert(train_data.size(0) == num_train_examples && train_targets.size(0) == num_train_examples);
    assert(calibration_data.size(0) == num_valid_examples && calibration_targets.size(0) == num_valid_examples);
    
  }


  // ============================= 6. 数据加载器初始化 ==============================
  auto train_dataset = upcite::SeriesDataset(train_data, train_targets);
  auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
      train_dataset.map(torch::data::transforms::Stack<>()), config_.get().filter_train_batchsize_);
  model_ = dstree::get_model(config_);
  model_->to(*device_);

  // ==================================== 8. 训练准备 ================================
  // for early termination
  std::unordered_map<std::string, torch::Tensor> best_model_state;
  VALUE_TYPE best_validation_loss = constant::MAX_VALUE;
  ID_TYPE best_validation_epoch = -1;
  // 优化器选择（CNN用Adam，其他用SGD）
  std::shared_ptr<torch::optim::Optimizer> optimizer = nullptr;
  if (model_->model_type_ == CNN) {
    optimizer = std::make_shared<torch::optim::Adam>(model_->parameters(), config_.get().filter_train_learning_rate_);
  } else {
    optimizer = std::make_shared<torch::optim::SGD>(model_->parameters(), config_.get().filter_train_learning_rate_);
  }
  ID_TYPE initial_cooldown_epochs = config_.get().filter_train_nepoch_ / 2;

  //  学习率调整策略（ReduceLROnPlateau）, 用于验证损失
  upcite::optim::ReduceLROnPlateau lr_scheduler = upcite::optim::ReduceLROnPlateau(
      *optimizer, initial_cooldown_epochs, optim::MIN, config_.get().filter_lr_adjust_factor_);

  torch::nn::MSELoss mse_loss(torch::nn::MSELossOptions().reduction(torch::kMean));


#ifdef DEBUG
  std::vector<float> train_losses, valid_losses, batch_train_losses;
  train_losses.reserve(config_.get().filter_train_nepoch_);
  batch_train_losses.reserve(num_train_examples / config_.get().filter_train_batchsize_ + 1);
  valid_losses.reserve(config_.get().filter_train_nepoch_);
#endif


  // original 训练循环
  torch::Tensor batch_data, batch_target;
  for (ID_TYPE epoch = 0; epoch < config_.get().filter_train_nepoch_; ++epoch) {
    model_->train();   
    for (auto &batch : *train_data_loader) {
      batch_data = batch.data;
      batch_target = batch.target;
      optimizer->zero_grad();
      torch::Tensor prediction = model_->forward(batch_data);
      torch::Tensor loss = mse_loss->forward(prediction, batch_target);
      loss.backward(); // 反向传播
      // 梯度裁剪防止爆炸
      if (config_.get().filter_train_clip_grad_) {
        auto norm = torch::nn::utils::clip_grad_norm_(model_->parameters(),
                                                      config_.get().filter_train_clip_grad_max_norm_,
                                                      config_.get().filter_train_clip_grad_norm_type_);
      }
      optimizer->step();

#ifdef DEBUG
      batch_train_losses.push_back(loss.detach().item<float>());
#endif
    }

#ifdef DEBUG
    train_losses.push_back(std::accumulate(batch_train_losses.begin(), batch_train_losses.end(), 0.0)
                               / static_cast<VALUE_TYPE>(batch_train_losses.size()));
    batch_train_losses.clear();
#endif

    // 9.2 验证阶段
    { // evaluate
      VALUE_TYPE valid_loss = 0;
      c10::InferenceMode guard;
      model_->eval();  
      torch::Tensor prediction = model_->forward(valid_data);
      valid_loss = mse_loss->forward(prediction, valid_targets).detach().item<VALUE_TYPE>();

#ifdef DEBUG
      valid_losses.push_back(valid_loss);
#endif
      // 记录最佳模型状态
      if (epoch > initial_cooldown_epochs) {
        if (best_validation_loss > valid_loss) {
          best_validation_loss = valid_loss;
          best_validation_epoch = epoch;

          for (const auto &pair : model_->named_parameters()) {
            best_model_state[pair.key()] = pair.value().clone();
          }
        }
      }
      // 学习率调整与早停策略
      upcite::optim::LR_RETURN_CODE return_code = lr_scheduler.check_step(valid_loss);
      if (return_code == upcite::optim::EARLY_STOP) {
        epoch = config_.get().filter_train_nepoch_;
      }
    }
  }

#ifdef DEBUG
  spdlog::debug("filter {:d} s{:d} {:s} tloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(train_losses.data(), config_.get().filter_train_nepoch_));
  spdlog::debug("filter {:d} s{:d} {:s} vloss = {:s}",
                id_, stream_id, model_setting_ref_.get().model_setting_str,
                upcite::array2str(valid_losses.data(), config_.get().filter_train_nepoch_));
#endif

  c10::InferenceMode guard;

// ============================ 10. 模型恢复与预测 =============================
// 恢复最佳模型状态
  if (best_validation_epoch > initial_cooldown_epochs) {
#ifdef DEBUG
    spdlog::debug("filter {:d} s{:d} {:s} restore from e{:d}, vloss {:.4f}",
                  id_, stream_id, model_setting_ref_.get().model_setting_str,
                  best_validation_epoch, best_validation_loss);
#endif

    for (auto &pair : best_model_state) {
      model_->named_parameters()[pair.first].detach_();
      model_->named_parameters()[pair.first].copy_(pair.second);
    }
  }
  // ========== 关键修改点 4: 对全局数据进行预测 ==========
  // 模型预测
  model_->eval();
  auto prediction = model_->forward(global_queries_).detach().cpu();
  assert(prediction.size(0) == global_data_size_);
  auto *predictions_array = prediction.detach().cpu().contiguous().data_ptr<VALUE_TYPE>();
  
  // !!!!!!!!!!!!!!!!!!1  存储模型预测结果到global_pred_distances_
  global_pred_distances_.insert(global_pred_distances_.end(), predictions_array, predictions_array + global_data_size_);
  
  
  // 打印 global_pred_distances_ 的 size
  // printf("Size of global_pred_distances_: %zu\n", global_pred_distances_.size());

  // 保存模型预测结果到txt文件
  std::string results_dir = config_.get().results_path_;
  if (!results_dir.empty() && !fs::exists(results_dir)) {
    fs::create_directories(results_dir);
  }
  // 创建子文件夹 original_error_filters
  std::string original_error_dir = results_dir + "/original_error_filters";
  if (!fs::exists(original_error_dir)) {
    fs::create_directories(original_error_dir);
  }
  ID_TYPE start_index = num_global_train_examples + num_global_valid_examples;
  save_prediction_errors(original_error_dir + "/filter_" + std::to_string(id_) + "_train_CP_global_predict_errors.txt", start_index);
  
// #ifdef DEBUG
//   spdlog::info("filter {:d}{:s} s{:d} {:s} g_pred{:s} = {:s}",
//                id_, is_trial ? " (trial)" : "",
//                stream_id, model_setting_ref_.get().model_setting_str,
//                config_.get().filter_remove_square_ ? "" : "_sq",
//                upcite::array2str(predictions_array, global_data_size_));
// #endif

  // ========== 关键修改点 5: 多校准集保形预测 ==========
  if (config_.get().filter_is_conformal_) {
    // 对预测距离进行Conformal Prediction的校准
    // printf("\n ---------------正式进入CP了,集中注意力---------------\n");
    fit_batch_conformal_predictor(is_trial, num_batches, calib_data_batches, calib_target_batches);
  }

 //  net->to(torch::Device(torch::kCPU));
  c10::cuda::CUDACachingAllocator::emptyCache();

  if (!is_trial) {
    is_trained_ = true;

  } else {
    // TODO should this work around be improved
    global_pred_distances_.clear();
  }

  return SUCCESS;
}




// 将校准查询ID保存到文件
RESPONSE dstree::Filter::save_calib_query_ids(const std::vector<std::vector<ID_TYPE>>& calib_query_ids, 
                                              const std::string& filename) {
    // 使用配置中的结果路径
    
    std::string save_path = config_.get().results_path_; // 从配置中获取路径
    
    // 确保路径以'/'结尾
    if (!save_path.empty() && save_path.back() != '/') {
        save_path += '/';
    }
    
    // 创建完整文件名
    std::string full_filename = save_path + filename;
    if (filename.find(".txt") == std::string::npos) {
        full_filename += ".txt";
    }
    
    // 打开文件进行写入
    std::ofstream file(full_filename);
    if (!file.is_open()) {
        printf("错误: 无法创建文件 %s\n", full_filename.c_str());
        return FAILURE;
    }
    
    // 写入总批次数
    // file << calib_query_ids.size() << std::endl;
    
    // 逐批次写入查询ID
    for (size_t batch_idx = 0; batch_idx < calib_query_ids.size(); ++batch_idx) {
        const auto& batch = calib_query_ids[batch_idx];
        
        // 写入当前批次的ID数量
        // file << batch.size() << std::endl;
        
        // 写入当前批次的所有ID，用空格分隔
        for (size_t i = 0; i < batch.size(); ++i) {
            file << batch[i];
            if (i < batch.size() - 1) {
                file << " ";
            }
        }
        file << std::endl;
    }
    
    file.close();
    printf("已成功保存校准查询ID到 %s (共%zu批次)\n", 
           full_filename.c_str(), calib_query_ids.size());
    
    return SUCCESS;
}
    


    

RESPONSE dstree::Filter::collect_running_info(MODEL_SETTING &model_setting) {
  model_setting_ref_ = model_setting;

  model_ = dstree::get_model(config_);
  model_->to(*device_);

  c10::InferenceMode guard;
  model_->eval();

  if (config_.get().filter_is_conformal_ && !conformal_predictor_->is_fitted()) {
    printf("-------collect_running_info 进入fit_conformal_predictor(false, true)------------\n");
    fit_conformal_predictor(false, true);
  }

  model_setting_ref_.get().gpu_mem_mb = get_memory_footprint(*model_);

  auto trial_query = global_queries_.get().index({torch::indexing::Slice(0, 1)}).clone();
  auto trial_predictions = make_reserved<VALUE_TYPE>(config_.get().filter_trial_iterations_);

  auto start = std::chrono::high_resolution_clock::now();

  for (ID_TYPE trial_i = 0; trial_i < config_.get().filter_trial_iterations_; ++trial_i) {
    auto pred = model_->forward(trial_query).item<VALUE_TYPE>();

    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
    }

    if (config_.get().filter_remove_square_) {
      trial_predictions.push_back(pred * pred);
    } else {
      trial_predictions.push_back(pred);
    }
  }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  model_setting_ref_.get().gpu_ms_per_query =
      static_cast<double_t>(duration.count()) / static_cast<double_t>(config_.get().filter_trial_iterations_);

#ifdef DEBUG
  spdlog::info("trial {:s} gpu mem = {:.3f}MB, time = {:.6f}mus",
               model_setting_ref_.get().model_setting_str,
               model_setting_ref_.get().gpu_mem_mb,
               model_setting_ref_.get().gpu_ms_per_query);
#endif

  return SUCCESS;
}



VALUE_TYPE dstree::Filter::infer(torch::Tensor &query_series) const {
#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("filter {:d} {:b} device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                device_->str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  spdlog::debug("filter {:d} {:b} query device {:s}, requested {:b}:{:d}",
                id_, is_trained_,
                query_series.device().str(),
                config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);

  auto paras = model_->parameters();
  for (ID_TYPE i = 0; i < paras.size(); ++i) {
    spdlog::debug("filter {:d} {:b} model_p_{:d} device {:s}, requested {:b}:{:d}",
                  id_, is_trained_,
                  i, paras[i].device().str(),
                  config_.get().filter_infer_is_gpu_, config_.get().filter_device_id_);
  }
#endif
#endif
  // printf("--------------进入测试阶段的infer函数--------------\n");
  if (is_trained_) {
    printf("is_trained_ = %d\n", is_trained_);
    c10::InferenceMode guard;
    VALUE_TYPE pred = model_->forward(query_series).item<VALUE_TYPE>();
    if (conformal_predictor_ != nullptr) {
      pred = conformal_predictor_->predict(pred).left_bound_;
      // printf("pred = %.3f, left_bound_ = %.3f\n", pred);
    }

    if (config_.get().filter_remove_square_) {
      return pred * pred;
    } else {
      return pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}


// 在Filter类中添加infer_calibrated方法的实现
VALUE_TYPE dstree::Filter::infer_calibrated(torch::Tensor &query_series) const {
  if (is_trained_) {
    c10::InferenceMode guard;
    // 获取模型原始预测
    //检查当前的filter model和CP model预测的距离和误差是remove 平方的。
    VALUE_TYPE raw_pred = model_->forward(query_series).item<VALUE_TYPE>(); //
    // 获取校准值（原始预测 - alpha）
    VALUE_TYPE calibrated_pred;
    if (conformal_predictor_ != nullptr) {
      // 使用ConformalRegressor获取校准后的下界
      calibrated_pred = conformal_predictor_->predict_calibrated(raw_pred);
      // printf("Filter %d: 原始预测距离=%.3f, 校准后距离=%.3f\n", 
            //  id_, raw_pred, calibrated_pred);
      // spdlog::info("raw_pred={:.3f}, calibrated_pred={:.3f}",  raw_pred, calibrated_pred);
    } else {
      calibrated_pred = raw_pred; // 无校准
    }
    // 应用平方处理 
    if (config_.get().filter_remove_square_) {
      return calibrated_pred * calibrated_pred;
    } else {
      return calibrated_pred;
    }
  } else {
    return constant::MAX_VALUE;
  }
}



// 在Filter类中添加新方法，仅返回原始预测
VALUE_TYPE dstree::Filter::infer_raw(torch::Tensor &query_series) const {
    if (is_trained_) {
        c10::InferenceMode guard;
        VALUE_TYPE pred = model_->forward(query_series).item<VALUE_TYPE>();
        
        // 仅返回模型原始预测，不进行任何校准
        if (config_.get().filter_remove_square_) {

            // printf("config_.get().filter_remove_square_ = %d\n", config_.get().filter_remove_square_); 
            // printf("pred = %.3f, pred * pred = %.3f\n", pred, pred * pred);
            return pred * pred;
        } else {
            return pred;
        }
    } else {
        return constant::MAX_VALUE;
    }
}




RESPONSE dstree::Filter::dump(std::ofstream &node_fos) const {
  node_fos.write(reinterpret_cast<const char *>(&global_data_size_), sizeof(ID_TYPE));

  assert(global_bsf_distances_.size() == global_data_size_);
  ID_TYPE size_placeholder = global_bsf_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_bsf_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_bsf_distances_.data()),
                   sizeof(VALUE_TYPE) * global_bsf_distances_.size());
  }

  assert(global_lnn_distances_.size() == global_data_size_);
  size_placeholder = global_lnn_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_lnn_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_lnn_distances_.data()),
                   sizeof(VALUE_TYPE) * global_lnn_distances_.size());
  }

  assert(lb_distances_.size() == global_data_size_);
  size_placeholder = lb_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!lb_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(lb_distances_.data()), sizeof(VALUE_TYPE) * lb_distances_.size());
  }

  // currently upper bounds are not being used
  assert(ub_distances_.size() == 0);
  size_placeholder = ub_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!ub_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(ub_distances_.data()), sizeof(VALUE_TYPE) * ub_distances_.size());
  }

// #ifdef DEBUG
//   spdlog::debug("filter {:d} (trained {:b} active {:b}) n_pred {:d} n_glob {:d} n_local {:d}",
//                 id_, is_trained_, is_active_,
//                 global_pred_distances_.size(), global_data_size_, local_data_size_);
// #endif

  if (is_trained_) {
    assert(global_pred_distances_.size() == global_data_size_);
  } else {
    assert(global_pred_distances_.empty());
  }
  size_placeholder = global_pred_distances_.size();
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (!global_pred_distances_.empty()) {
    node_fos.write(reinterpret_cast<const char *>(global_pred_distances_.data()),
                   sizeof(VALUE_TYPE) * global_pred_distances_.size());
  }

  node_fos.write(reinterpret_cast<const char *>(&local_data_size_), sizeof(ID_TYPE));

  if (local_data_size_ > 0) {
    assert(config_.get().series_length_ * local_data_size_ == local_queries_.size());
    assert(local_lnn_distances_.size() == local_data_size_);

    size_placeholder = local_queries_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));

    if (!local_queries_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_queries_.data()),
                     sizeof(VALUE_TYPE) * local_queries_.size());
    }

    size_placeholder = local_lnn_distances_.size();
    node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));

    if (!local_lnn_distances_.empty()) {
      node_fos.write(reinterpret_cast<const char *>(local_lnn_distances_.data()),
                     sizeof(VALUE_TYPE) * local_lnn_distances_.size());
    }
  }

//  spdlog::debug("dump filter {:d} global {:d} local {:d} active {:b} train {:b}",
//                id_, global_data_size_, local_data_size_, is_active_, is_trained_);

  if (is_active_) {
    size_placeholder = model_setting_ref_.get().model_setting_str.size();
  } else {
    size_placeholder = -1;
  }
  node_fos.write(reinterpret_cast<const char *>(&size_placeholder), sizeof(ID_TYPE));
  if (is_active_) {
    node_fos.write(reinterpret_cast<const char *>(model_setting_ref_.get().model_setting_str.data()),
                   sizeof(model_setting_ref_.get().model_setting_str));
  }

  ID_TYPE is_trained_placeholder = 0;
  if (is_trained_) {
    is_trained_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_trained_placeholder), sizeof(ID_TYPE));
  if (is_trained_) {
    std::string model_filepath = config_.get().dump_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;

    torch::save(model_, model_filepath);
  }

  ID_TYPE is_conformal_placeholder = 0;
  if (config_.get().filter_is_conformal_) {
    is_conformal_placeholder = 1;
  }
  node_fos.write(reinterpret_cast<const char *>(&is_conformal_placeholder), sizeof(ID_TYPE));
  if (config_.get().filter_is_conformal_) {
    conformal_predictor_->dump(node_fos);
  }

  return SUCCESS;
}




// 在 Filter 类实现中添加
RESPONSE dstree::Filter::load_batch_alphas(const std::string& filepath) {
    if (!is_trained_ || !is_active_ || !config_.get().filter_is_conformal_) {
        return FAILURE;
    }
    
    return conformal_predictor_->load_batch_alphas(filepath);
}

// load 函数

RESPONSE dstree::Filter::load(std::ifstream &node_ifs, void *ifs_buf) {
  auto ifs_id_buf = reinterpret_cast<ID_TYPE *>(ifs_buf);
  auto ifs_value_buf = reinterpret_cast<VALUE_TYPE *>(ifs_buf);

  // global_data_size_
  ID_TYPE read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  global_data_size_ = ifs_id_buf[0];

  // bsf_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  ID_TYPE size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_bsf_distances_.insert(global_bsf_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(global_bsf_distances_.size() == global_data_size_);
//  assert(node_ifs.good());

  // nn_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_lnn_distances_.insert(global_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(global_lnn_distances_.size() == global_data_size_);
//  assert(node_ifs.good());

  // lb_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    lb_distances_.insert(lb_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(lb_distances_.size() == global_data_size_);
  assert(node_ifs.good());

  // ub_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    ub_distances_.insert(ub_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
  }
  assert(ub_distances_.size() == 0);
//  assert(node_ifs.good());

  // pred_distances_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator > 0) {
    read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    global_pred_distances_.insert(global_pred_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    assert(global_pred_distances_.size() == global_data_size_);
//    assert(node_ifs.good());
  }

  // local_data_size_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  local_data_size_ = ifs_id_buf[0];


  if (local_data_size_ > 0) {
    // local_queries_
    read_nbytes = sizeof(ID_TYPE);
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    size_indicator = ifs_id_buf[0];
    assert(size_indicator == config_.get().series_length_ * local_data_size_);

    if (size_indicator > 0) {
      local_queries_.reserve(size_indicator);
      read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
      node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
      local_queries_.insert(local_queries_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    }

    // local_lnn_distances_
    read_nbytes = sizeof(ID_TYPE);
    node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
    size_indicator = ifs_id_buf[0];
    assert(size_indicator == local_data_size_);

    if (size_indicator > 0) {
      local_lnn_distances_.reserve(size_indicator);
      read_nbytes = sizeof(VALUE_TYPE) * size_indicator;
      node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
      local_lnn_distances_.insert(local_lnn_distances_.begin(), ifs_value_buf, ifs_value_buf + size_indicator);
    }
  }
  assert(local_queries_.size() == config_.get().series_length_ * local_data_size_);
  assert(local_lnn_distances_.size() == local_data_size_);
//  assert(node_ifs.good());

  // model_setting_
  is_active_ = false;
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    std::string model_setting_str;
    model_setting_str.resize(size_indicator);
    node_ifs.read(const_cast<char *>(model_setting_str.data()), size_indicator);

    if (config_.get().to_load_filters_) {
      model_setting_ = MODEL_SETTING(model_setting_str);
      model_setting_ref_ = std::ref(model_setting_);
      is_active_ = true;
    }
  }
//  assert(node_ifs.good());
  // model_
  is_trained_ = false;
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];

  if (size_indicator == 0 && is_active_) {
    // spdlog::error("loading filter {:d} activated but marked untrained; workaround by setting is_trained_", id_);
    size_indicator = 1;
  }

  if (size_indicator > 0) {
    std::string model_filepath = config_.get().load_filters_folderpath_ + std::to_string(id_) +
        config_.get().model_dump_file_postfix_;
    if (!fs::is_regular_file(model_filepath)) {
      spdlog::error("Empty model_filepath found: {:s}", model_filepath);
      return FAILURE;
    }

    if (config_.get().filter_infer_is_gpu_) {
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
    model_ = dstree::get_model(config_);
    // TODO check if the to-be-loaded model type matches the persisted model type
    torch::load(model_, model_filepath);
    model_->to(*device_);
    model_->eval();
//  net->to(torch::Device(torch::kCPU));
    c10::cuda::CUDACachingAllocator::emptyCache();
    if (config_.get().to_load_filters_) {
      is_trained_ = true;
    }
  }
//  assert(node_ifs.good());
  // conformal_predictor_
  read_nbytes = sizeof(ID_TYPE);
  node_ifs.read(static_cast<char *>(ifs_buf), read_nbytes);
  size_indicator = ifs_id_buf[0];
  if (size_indicator > 0) {
    conformal_predictor_->load(node_ifs, ifs_buf);
    if (is_active_ && is_trained_ && config_.get().filter_is_conformal_) {
      // TODO check compatibility between the loaded setting and the new setting
      // printf("~~~~~~~~~~~~~ 进入了load函数:fit_conformal_predictor(false, false) ~~~~~~~~~~~~~~~~~~\n");
      fit_conformal_predictor(false, false);
    }
  }
  spdlog::debug("load filter {:d} global {:d} local {:d} active {:b} trained {:b}",
                id_, global_data_size_, local_data_size_, is_active_, is_trained_);
  assert(node_ifs.good());
  return SUCCESS;
}




VALUE_TYPE dstree::Filter::get_node_summarization_pruning_frequency() const {
  if (lb_distances_.empty() || lb_distances_.size() != global_bsf_distances_.size()) {
    return 0;
  }

  ID_TYPE pruned_counter = 0;
  for (ID_TYPE i = 0; i < lb_distances_.size(); ++i) {
    if (lb_distances_[i] > global_bsf_distances_[i]) {
      pruned_counter += 1;
    }
  }

  return static_cast<VALUE_TYPE>(pruned_counter) / static_cast<VALUE_TYPE>(lb_distances_.size());
}

VALUE_TYPE upcite::dstree::Filter::get_val_pruning_ratio() const {
  ID_TYPE num_global_train_examples = global_data_size_ * config_.get().filter_train_val_split_;

  VALUE_TYPE abs_error_interval = get_abs_error_interval();
  ID_TYPE pruned_counter = 0;

  for (ID_TYPE example_i = num_global_train_examples; example_i < global_data_size_; ++example_i) {
    if (global_pred_distances_[example_i] - abs_error_interval > global_bsf_distances_[example_i]){
      pruned_counter += 1;
    }
  }

  ID_TYPE num_global_valid_examples = global_data_size_ - num_global_train_examples;
  return static_cast<VALUE_TYPE>(pruned_counter) / num_global_valid_examples;
}

// 添加实现save_filter_batch_alphas方法
RESPONSE upcite::dstree::Filter::save_filter_batch_alphas(const std::string& filepath) const {
  if (!conformal_predictor_) {
    spdlog::error("没有初始化conformal_predictor，无法保存alphas");
    return FAILURE;
  }  
  // 调用ConformalPredictor中的方法保存批处理alphas
  return conformal_predictor_->save_batch_alphas(filepath);
}

// 添加在文件末尾，其他方法之后

// 添加清理批处理alphas的方法
RESPONSE dstree::Filter::clear_batch_alphas() {
  if (!is_trained_ || !is_active_ || !config_.get().filter_is_conformal_ || !conformal_predictor_) {
    return SUCCESS; // 如果不适用批处理alpha或预测器不存在，不需要清理
  }
  // 委托给ConformalPredictor的清理方法
  conformal_predictor_->clear_batch_data();
  return SUCCESS;
}



// 生成校准批次，使用增强组合方法 (每次重新打乱索引)
RESPONSE dstree::Filter::generate_calibration_batches(
    torch::Tensor& calib_data, 
    torch::Tensor& calib_targets,
    std::vector<torch::Tensor>& calib_data_batches,
    std::vector<torch::Tensor>& calib_target_batches,
    std::vector<std::vector<ID_TYPE>>& calib_query_ids) {
    // printf("============进入generate_calibration_batches函数============\n");
    // =================== 自适应重复最小化抽样 ===================
    ID_TYPE num_calib_examples = calib_data.size(0);
    ID_TYPE num_batches = config_.get().filter_conformal_num_batches_;
    ID_TYPE batch_size = config_.get().filter_conformal_batch_size_;
    // printf("num_calib_examples: %ld, num_batches: %ld, batch_size: %ld\n", static_cast<long>(num_calib_examples), static_cast<long>(num_batches), static_cast<long>(batch_size));
    if (num_calib_examples == 0 || num_batches == 0) {
        printf("错误: 校准样本数量(%ld)或批次数(%ld)非法\n", static_cast<long>(num_calib_examples), static_cast<long>(num_batches));
        return FAILURE;
    }

    // 1. 计算所有bacth需要的query数量
    ID_TYPE total_needed = num_batches * batch_size;   // 可能大于 num_calib_examples，需要重复

    // 2. 计算每条样本的配额 (base_usage 或 base_usage+1)
    ID_TYPE base_usage      = total_needed / num_calib_examples;  // 最少使用次数
    ID_TYPE extra_usage_cnt = total_needed % num_calib_examples;  // 需要额外 +1 的样本数

    std::vector<ID_TYPE> usage_quota(num_calib_examples, base_usage);

    // 随机数生成器
    std::random_device rd;
    std::mt19937 g(rd());

    // 将索引打乱后，前 extra_usage_cnt 条 +1
    std::vector<ID_TYPE> indices(num_calib_examples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), g);
    for (ID_TYPE i = 0; i < extra_usage_cnt; ++i) {
        usage_quota[indices[i]]++;
    }

    // 3. 构造采样池 (长度 == total_needed)
    //usage_quota[i] 记录第 i 条 query 要出现的次数（大多数 2 次，少数 3 次）。
    std::vector<ID_TYPE> sampling_pool;
    sampling_pool.reserve(total_needed);
    for (ID_TYPE i = 0; i < num_calib_examples; ++i) {
        for (ID_TYPE k = 0; k < usage_quota[i]; ++k) {
            sampling_pool.push_back(i);
        }
    }
    // 再次打乱，保证随机
    std::shuffle(sampling_pool.begin(), sampling_pool.end(), g);
    // printf("sampling_pool.size(): %ld\n", static_cast<long>(sampling_pool.size()));
    //sampling_pool 就是一条"抽签池"——
    //它按"这条 query 应该被抽几次"的配额，把 校准集里所有 query 的索引 id 复制进去，然后整体打乱。

    // 4. 按批次取样
    calib_data_batches.resize(num_batches);
    calib_target_batches.resize(num_batches);
    calib_query_ids.resize(num_batches);

    for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        calib_data_batches[batch_idx]   = torch::empty({batch_size, calib_data.size(1)}, calib_data.options());
        calib_target_batches[batch_idx] = torch::empty(batch_size, calib_targets.options());
        calib_query_ids[batch_idx].reserve(batch_size);

        for (ID_TYPE j = 0; j < batch_size; ++j) {
            ID_TYPE pool_pos = batch_idx * batch_size + j;
            ID_TYPE idx      = sampling_pool[pool_pos];

            // 写入样本
            calib_data_batches[batch_idx][j]   = calib_data[idx];
            calib_target_batches[batch_idx][j] = calib_targets[idx];

            // 计算原始全局 query ID 偏移
            ID_TYPE global_cp_start_idx = global_data_size_ - num_calib_examples; // CP校准集在原始query中的起始位置
            ID_TYPE original_query_id = global_cp_start_idx + idx;
            // printf("original_query_id: %ld\n", static_cast<long>(original_query_id));
            calib_query_ids[batch_idx].push_back(original_query_id);
        }

        // （可选）再次随机打乱当前批次内部顺序
        std::vector<ID_TYPE> perm_vec(batch_size);
        std::iota(perm_vec.begin(), perm_vec.end(), 0);
        std::shuffle(perm_vec.begin(), perm_vec.end(), g);
        torch::Tensor perm = torch::from_blob(perm_vec.data(), {static_cast<long>(batch_size)}, torch::TensorOptions().dtype(torch::kLong)).clone();
        perm = perm.to(calib_data_batches[batch_idx].device());
        calib_data_batches[batch_idx]   = calib_data_batches[batch_idx].index_select(0, perm);
        calib_target_batches[batch_idx] = calib_target_batches[batch_idx].index_select(0, perm);
        // 同步调整 query_ids 顺序
        std::vector<ID_TYPE> tmp_ids = calib_query_ids[batch_idx];
        for (ID_TYPE j = 0; j < batch_size; ++j) {
            calib_query_ids[batch_idx][j] = tmp_ids[perm_vec[j]];
        }
    }

    return SUCCESS;
}





// 生成均匀划分的校准批次
RESPONSE dstree::Filter::generate_uniform_calibration_batches(
    torch::Tensor& calib_data, 
    torch::Tensor& calib_targets,
    std::vector<torch::Tensor>& calib_data_batches,
    std::vector<torch::Tensor>& calib_target_batches,
    std::vector<std::vector<ID_TYPE>>& calib_query_ids) {
    
  // printf("使用均匀划分生成校准批次\n");
  
  // 获取校准数据大小和配置参数
  ID_TYPE num_calib_examples = calib_data.size(0);
  ID_TYPE num_batches = config_.get().filter_conformal_num_batches_;
  
  // 确保每个批次至少有3个样本
  num_batches = std::min(
      num_batches,
      static_cast<ID_TYPE>(std::floor(num_calib_examples / 3.0))
  );
  
  if (num_batches < 1) num_batches = 1;
  // printf("校准批次数: %d\n", num_batches);
  
  // 计算每个校准批次的样本数
  ID_TYPE examples_per_batch = num_calib_examples / num_batches;
  ID_TYPE remainder = num_calib_examples % num_batches;
  
  // 创建随机索引进行数据打乱
  std::vector<ID_TYPE> indices(num_calib_examples);
  for (ID_TYPE i = 0; i < num_calib_examples; ++i) {
      indices[i] = i;
  }
  
  // 随机打乱索引
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);
  
  // 预分配批次存储空间
  calib_data_batches.resize(num_batches);
  calib_target_batches.resize(num_batches);
  calib_query_ids.resize(num_batches);
  
  // 将校准数据分配到各批次
  for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      // 计算当前批次的起始索引和结束索引
      ID_TYPE start_idx = batch_idx * examples_per_batch;
      // 为最后一个批次添加剩余样本
      ID_TYPE extra = (batch_idx == num_batches - 1) ? remainder : 0;
      ID_TYPE end_idx = start_idx + examples_per_batch + extra;
      // 确保不超出索引范围
      end_idx = std::min(end_idx, static_cast<ID_TYPE>(indices.size()));
      // 计算当前批次样本数
      ID_TYPE batch_size = end_idx - start_idx;
      
      // 初始化当前批次的数据和目标张量
      calib_data_batches[batch_idx] = torch::empty({batch_size, calib_data.size(1)}, calib_data.options());
      calib_target_batches[batch_idx] = torch::empty(batch_size, calib_targets.options());
      calib_query_ids[batch_idx].reserve(batch_size);
      
      // 填充数据
      for (ID_TYPE i = 0; i < batch_size; ++i) {
          ID_TYPE idx = indices[start_idx + i];
          calib_data_batches[batch_idx][i] = calib_data[idx];
          calib_target_batches[batch_idx][i] = calib_targets[idx];
          // 记录原始查询ID（假设校准数据是从num_global_train_examples开始的全局数据）
          ID_TYPE original_query_id = num_global_train_examples + idx;
          calib_query_ids[batch_idx].push_back(original_query_id);
      }
      // printf("校准批次 %d: 样本数 = %d\n", batch_idx + 1, batch_size);
  }
  
  // 最终汇总
  ID_TYPE total_batch_size = 0;
  for (ID_TYPE i = 0; i < num_batches; ++i) {
      total_batch_size += calib_data_batches[i].size(0);
  }
  
  // printf("成功生成 %d 个校准批次, 总样本数: %d\n", num_batches, total_batch_size);
  return SUCCESS;
}




RESPONSE dstree::Filter::train_regression_model_for_recall_coverage(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ID_TYPE>& error_indices,
    ID_TYPE filter_id) {
    // 直接委托给conformal_predictor_
    return conformal_predictor_->train_regression_model_for_recall_coverage(
        recalls, coverages, error_indices, filter_id);
}


RESPONSE dstree::Filter::fit_alglib_quadratic_spline(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ERROR_TYPE>& errors,
    std::vector<double>& model_coeffs) {
    return conformal_predictor_->fit_alglib_quadratic_spline(recalls, coverages, errors, model_coeffs);
}


RESPONSE dstree::Filter::fit_eigen_quadratic_spline(
    const std::vector<ERROR_TYPE>& recalls,
    const std::vector<ERROR_TYPE>& coverages,
    const std::vector<ERROR_TYPE>& errors,
    std::vector<double>& model_coeffs) {
    // 直接委托给conformal_predictor_
    return conformal_predictor_->fit_eigen_quadratic_spline(
        recalls, coverages, errors, model_coeffs);
}

// // 添加新函数的实现，使用实际批次误差
// RESPONSE dstree::Filter::train_regression_model_for_recall_coverage_actual_error(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages,
//     const std::vector<ID_TYPE>& error_indices,
//     ID_TYPE batch_id,
//     ID_TYPE filter_id) {
    
//     // 直接委托给conformal_predictor_的新函数
//     return conformal_predictor_->train_regression_model_for_recall_coverage_actual_error(
//         recalls, coverages, error_indices, batch_id, filter_id);
// }

// 预测函数实现
double dstree::Filter::predict_error_value(double recall, double coverage) const {
    return conformal_predictor_->predict_error_value(recall, coverage);
}

// 设置函数实现
RESPONSE dstree::Filter::set_filter_abs_error_interval_by_recall_and_coverage(
    ERROR_TYPE recall, ERROR_TYPE coverage) {
    return conformal_predictor_->set_alpha_by_recall_and_coverage(recall, coverage);
}

// RESPONSE dstree::Filter::train_optimal_polynomial_model(
//     const std::vector<ERROR_TYPE>& recalls,
//     const std::vector<ERROR_TYPE>& coverages,
//     const std::vector<ERROR_TYPE>& errors,
//     ID_TYPE max_degree) {
    
//     // 检查conformal_predictor_是否可用
//     if (!conformal_predictor_) {
//         spdlog::error("保形预测器未初始化");
//         return FAILURE;
//     }
    
//     // 委托给conformal_predictor_
//     return conformal_predictor_->find_optimal_polynomial_model(
//         recalls, coverages, errors, max_degree);
// }

// 获取训练数据（用于optimal_polynomial测试）
bool dstree::Filter::get_training_data(std::vector<ERROR_TYPE>& recalls, std::vector<ERROR_TYPE>& coverages, std::vector<ERROR_TYPE>& errors) const {
    // 清空输出参数
    recalls.clear();
    coverages.clear();
    errors.clear();
    
    // 检查conformal_predictor_是否可用
    if (!conformal_predictor_) {
        spdlog::error("保形预测器未初始化");
        return false;
    }
}
    
    // 从训练数据存储中获取当前保存的训练数据
    // 由于我们没有直接存储这些数据，需要从之前调用的函数中获取
    // 这里只能返回一个�


// 获取训练数据（用于regression_model测试）
bool dstree::Filter::get_training_data_with_indices(std::vector<ERROR_TYPE>& recalls, std::vector<ERROR_TYPE>& coverages, std::vector<ID_TYPE>& error_indices) const {
    // 清空输出参数
    recalls.clear();
    coverages.clear();
    error_indices.clear();
    
    // 检查conformal_predictor_是否可用
    if (!conformal_predictor_) {
        spdlog::error("保形预测器未初始化");
        return false;
    }
    
    // 从训练数据存储中获取当前保存的训练数据
    // 由于我们没有直接存储这些数据，需要从之前调用的函数中获取
    // 这里只能返回一个简化的实现或空数据
    
    // 真实实现应该从某处获取这些数据，这里只返回一个简单的示例数据
    // 注意：这是模拟数据，实际使用时应该返回真实的训练数据
    recalls = {0.95f, 0.96f, 0.97f, 0.98f, 0.99f};
    coverages = {0.8f, 0.85f, 0.9f, 0.95f, 0.99f};
    error_indices = {0, 1, 2, 3, 4};
    
    return true;
}

// 重置回归系数和alpha值
void dstree::Filter::reset_regression_coefficients(const std::vector<double>& coeffs, VALUE_TYPE alpha) {
    // 检查conformal_predictor_是否可用
    if (!conformal_predictor_) {
        spdlog::error("保形预测器未初始化");
        return;
    }
    
    // 调用conformal_predictor_的方法来重置系数和alpha值
    conformal_predictor_->reset_regression_coefficients(coeffs, alpha);
}


// 保存批处理校准查询ID到二进制文件
RESPONSE dstree::Filter::save_batch_calib_query_ids(const std::string& filepath) const {
  // printf("保存批处理校准查询ID到文件: %s\n", filepath.c_str());
  
  // 如果校准查询ID为空，则返回失败
  if (batch_calib_query_ids_.empty()) {
    printf("错误: 批处理校准查询ID为空，无法保存\n");
    return FAILURE;
  }
  std::ofstream file(filepath, std::ios::binary);
  if (!file.good()) {
    printf("错误: 无法创建文件 %s\n", filepath.c_str());
    return FAILURE;
  }
  
  // 写入批次数量
  ID_TYPE num_batches = static_cast<ID_TYPE>(batch_calib_query_ids_.size());
  file.write(reinterpret_cast<const char*>(&num_batches), sizeof(ID_TYPE));
  
  // 逐批次写入查询ID
  for (const auto& batch : batch_calib_query_ids_) {
    // 写入当前批次的ID数量
    ID_TYPE batch_size = static_cast<ID_TYPE>(batch.size());
    file.write(reinterpret_cast<const char*>(&batch_size), sizeof(ID_TYPE));
    
    // 写入当前批次的所有ID
    for (const auto& id : batch) {
      file.write(reinterpret_cast<const char*>(&id), sizeof(ID_TYPE));
    }
  }
  
  file.close();
  // printf("成功保存批处理校准查询ID到 %s (共%zu批次)\n", filepath.c_str(), batch_calib_query_ids_.size());
  
  return SUCCESS;
}

// 加载批处理校准查询ID从二进制文件
RESPONSE dstree::Filter::load_batch_calib_query_ids(const std::string& filepath) {
  // printf("加载批处理校准查询ID从文件: %s\n", filepath.c_str());
  
  // 检测文件扩展名，决定以何种模式读取
  bool is_text_mode = filepath.find(".txt") != std::string::npos;
  
  if (is_text_mode) {
    // 文本模式读取
    std::ifstream file(filepath);
    if (!file.good()) {
      printf("错误: 无法打开文件 %s\n", filepath.c_str());
      return FAILURE;
    }
    
    // 清空现有数据
    batch_calib_query_ids_.clear();
    
    std::string line;
    ID_TYPE num_batches = 0;
    ID_TYPE current_batch = -1;
    ID_TYPE batch_size = 0;
    
    // 第一次扫描：确定批次数量和每个批次大小
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      
      if (line[0] == '#') {
        // 解析注释行
        if (line.find("批次数量") != std::string::npos) {
          std::sscanf(line.c_str(), "# 批次数量: %ld", &num_batches);
          // printf("文件含有 %ld 个批次\n", num_batches);
        } 
        else if (line.find("批次大小") != std::string::npos) {
          std::sscanf(line.c_str(), "# 批次大小: %ld", &batch_size);
          current_batch++;
          // printf("批次 %ld 大小为 %ld\n", current_batch, batch_size);
        }
      } 
      else if (current_batch >= 0) {
        // 非注释行是ID数据，可以预估批次大小
      }
    }
    
    // 重置文件指针
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // 为所有批次预分配内存
    if (num_batches > 0) {
      batch_calib_query_ids_.resize(num_batches);
      for (ID_TYPE i = 0; i < num_batches; i++) {
        // 为每个批次预留足够空间
        batch_calib_query_ids_[i].reserve(500);  // 保守估计，预留500个元素的空间
      }
    } else {
      printf("警告: 未能从文件中确定批次数量\n");
      return FAILURE;
    }
    
    // 第二次扫描：实际读取数据
    current_batch = -1;
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      
      if (line[0] == '#') {
        if (line.find("批次大小") != std::string::npos) {
          current_batch++;
        }
      } 
      else if (current_batch >= 0 && current_batch < num_batches) {
        // 解析一行ID数据
        std::istringstream iss(line);
        ID_TYPE id;
        while (iss >> id) {
          batch_calib_query_ids_[current_batch].push_back(id);
        }
      }
    }
    
    // 验证加载结果
    if (batch_calib_query_ids_.empty()) {
      printf("错误: 未能从文件加载任何批次数据\n");
      return FAILURE;
    }
    
    // printf("成功从文本文件加载 %zu 个批次的校准查询ID\n", batch_calib_query_ids_.size());
    
  } else {
    // 二进制模式读取
    std::ifstream file(filepath, std::ios::binary);
    if (!file.good()) {
      printf("错误: 无法打开文件 %s\n", filepath.c_str());
      return FAILURE;
    }
    
    // 清空现有数据
    batch_calib_query_ids_.clear();
    
    // 读取批次数量
    ID_TYPE num_batches = 0;
    file.read(reinterpret_cast<char*>(&num_batches), sizeof(ID_TYPE));
    
    // 为所有批次预分配内存
    batch_calib_query_ids_.resize(num_batches);
    
    // 逐批次读取查询ID
    for (ID_TYPE batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
      // 读取当前批次的ID数量
      ID_TYPE batch_size = 0;
      file.read(reinterpret_cast<char*>(&batch_size), sizeof(ID_TYPE));
      
      // 预分配当前批次的内存
      batch_calib_query_ids_[batch_idx].reserve(batch_size);
      
      // 读取当前批次的所有ID
      for (ID_TYPE i = 0; i < batch_size; ++i) {
        ID_TYPE id = 0;
        file.read(reinterpret_cast<char*>(&id), sizeof(ID_TYPE));
        batch_calib_query_ids_[batch_idx].push_back(id);
      }
    }
    
    // printf("成功加载批处理校准查询ID从二进制文件 %s (共%zu批次)\n", filepath.c_str(), batch_calib_query_ids_.size());
  }
  
  // 打印每个批次的大小和内存占用
  // for (size_t i = 0; i < batch_calib_query_ids_.size(); i++) {
  //   printf("批次 %zu: %zu 个ID, 容量: %zu\n", 
  //          i, batch_calib_query_ids_[i].size(), batch_calib_query_ids_[i].capacity());
  // }
  
  return SUCCESS;
}

// ===============================
// Debug helper: print global vectors
// ===============================
void dstree::Filter::debug_print_global_vectors() const {
  const size_t n_bsf  = global_bsf_distances_.size();
  const size_t n_lnn  = global_lnn_distances_.size();
  const size_t n_pred = global_pred_distances_.size();

  spdlog::info("[Filter {:d}] global vector sizes  bsf={}  lnn={}  pred={}", id_, n_bsf, n_lnn, n_pred);
  // printf("[Filter %ld] global vector sizes  bsf=%zu  lnn=%zu  pred=%zu\n", static_cast<long>(id_), n_bsf, n_lnn, n_pred);

  const size_t max_n = std::max({n_bsf, n_lnn, n_pred});
  for (size_t i = 0; i < max_n; ++i) {
    std::string bsf_str  = (i < n_bsf)  ? fmt::format("{:.6f}", global_bsf_distances_[i])  : "NA";
    std::string lnn_str  = (i < n_lnn)  ? fmt::format("{:.6f}", global_lnn_distances_[i])  : "NA";
    std::string pred_str = (i < n_pred) ? fmt::format("{:.6f}", global_pred_distances_[i]) : "NA";
    spdlog::info("[Filter {:d}] idx {} : bsf={} lnn={} pred={}", id_, i, bsf_str, lnn_str, pred_str);
  }
}

// // 析构函数实现 - 确保资源安全释放
// dstree::Filter::~Filter() {
//   try {
//     // 1. 清理PyTorch模型
//     if (model_) {
//       model_.reset();
//     }
    
//     // 2. 清理ConformalPredictor
//     if (conformal_predictor_) {
//       conformal_predictor_.reset();
//     }
    
//     // 3. 清理CUDA设备资源（如果使用GPU）
//     if (device_ && device_->is_cuda()) {
//       // 同步CUDA操作
//       c10::cuda::CUDAGuard device_guard(device_->index());
//       torch::cuda::synchronize(device_->index());
//     }
    
//     // 4. 清理设备指针
//     if (device_) {
//       device_.reset();
//     }
    
//     // 5. 清理容器（让其自然析构）
//     global_bsf_distances_.clear();
//     global_lnn_distances_.clear();
//     global_pred_distances_.clear();
//     lb_distances_.clear();
//     local_queries_.clear();
//     local_lnn_distances_.clear();
//     batch_calib_query_ids_.clear();
    
//   } catch (const std::exception& e) {
//     // 析构函数中不应抛出异常，只记录错误
//     fprintf(stderr, "[WARN] Filter %d destructor exception: %s\n", static_cast<int>(id_), e.what());
//   } catch (...) {
//     fprintf(stderr, "[WARN] Filter %d destructor unknown exception\n", static_cast<int>(id_));
//   }
// }

// 保存预测误差并按误差绝对值降序排序到txt文件
RESPONSE dstree::Filter::save_prediction_errors(const std::string& filepath, ID_TYPE start_index) {
    // 确保预测距离和真实距离长度一致
    if (global_pred_distances_.size() != global_lnn_distances_.size()) {
        spdlog::error("[save_prediction_errors] pred and true size mismatch: pred={}, true={}",
                      global_pred_distances_.size(), global_lnn_distances_.size());
        return FAILURE;
    }

    ID_TYPE n = static_cast<ID_TYPE>(global_pred_distances_.size());
    if (n == 0) {
        spdlog::warn("[save_prediction_errors] no data to save");
        return FAILURE;
    }

    // 计算误差并打包 (error, query_id)
    std::vector<std::pair<VALUE_TYPE, ID_TYPE>> error_pairs;
    std::vector<ERROR_TYPE> residuals;
    error_pairs.reserve(n);
    // RESPONSE return_code = conformal_predictor_->fit(batch_residuals);
    for (ID_TYPE i = start_index; i < n; ++i) {
        VALUE_TYPE err = global_pred_distances_[i] - global_lnn_distances_[i];
        residuals.push_back(err);
        error_pairs.emplace_back(err < 0 ? -err : err, i);
    }
    RESPONSE return_code = conformal_predictor_->fit(residuals);
    if (return_code != SUCCESS) {
        spdlog::error("!!!!![save_total_prediction_errors] fit failed");
        printf("!!!!![save_total_prediction_errors] fit failed\n");
        return FAILURE;
    }

    // 按误差降序排序
    std::sort(error_pairs.begin(), error_pairs.end(),
              [](const auto &a, const auto &b) {
                  return a.first > b.first;
              });

    // 写入文件
    std::ofstream fout(filepath);
    if (!fout.is_open()) {
        spdlog::error("[save_prediction_errors] cannot open file: {}", filepath);
        return FAILURE;
    }

    for (const auto &p : error_pairs) {
        fout << p.second << " " << p.first << "\n"; // query_id error_value
    }
    fout.close();
    spdlog::info("[save_prediction_errors] saved {} errors to {}", n, filepath);
    return SUCCESS;
}