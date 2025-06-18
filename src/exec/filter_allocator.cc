//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#include "filter_allocator.h"

#include <random>
#include <immintrin.h>
#include <queue>
#include <stack>
#include <utility>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/fmt.h>  // for fmt::format
#include <torch/torch.h>
#include <cuda_runtime_api.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <boost/filesystem.hpp>

#include "vec.h"
#include "distance.h"
#include "answer.h"
#include "index.h"
namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

// 添加必要的命名空间引用
namespace fs = boost::filesystem;

dstree::Allocator::Allocator(dstree::Config &config, 
                             //  const std::vector<dstree::Answers>& train_answers, // 新增参数
                             ID_TYPE nfilters) :
    config_(config),
    // train_answers_(train_answers),
    is_recall_calculated_(false),
    node_size_threshold_(0),
    min_validation_recall_(1) {

  if (config_.get().filter_infer_is_gpu_) {
    if (torch::cuda::is_available()) {
      cudaSetDevice(config_.get().filter_device_id_);

      size_t gpu_free_bytes_, gpu_total_bytes_;
      cudaMemGetInfo(&gpu_free_bytes_, &gpu_total_bytes_);
      VALUE_TYPE gpu_free_mb = static_cast<VALUE_TYPE>(gpu_free_bytes_) / 1024 / 1024;

      if (gpu_free_mb < config.filter_max_gpu_memory_mb_) {
        if (gpu_free_mb > 1) {
          spdlog::error("allocator required {:.3f}mb is not available; down to all free {:.3f}mb",
                        config.filter_max_gpu_memory_mb_, gpu_free_mb);

          available_gpu_memory_mb_ = gpu_free_mb;
        } else {
          spdlog::error("allocator only {:.3f}mb gpu memory is free; exit", gpu_free_mb);
          spdlog::shutdown();
          exit(FAILURE);
        }
      } else {
        spdlog::info("allocator requested {:.3f}mb; {:.3f}mb available",
                     config.filter_max_gpu_memory_mb_, gpu_free_mb);

        available_gpu_memory_mb_ = config.filter_max_gpu_memory_mb_;
      }
    } else {
      spdlog::error("allocator gpu unavailable");
      spdlog::shutdown();
      exit(FAILURE);
    }
  }

  // TODO support model setting list
  candidate_model_settings_.emplace_back(config.filter_model_setting_str_);

  if (nfilters > 0) {
    filter_infos_.reserve(nfilters);
  }

  measure_cpu();
  assert(config_.get().filter_infer_is_gpu_);
  measure_gpu();

  node_size_threshold_ = constant::MAX_ID;
  for (ID_TYPE candidate_model_i = 0; candidate_model_i < candidate_model_settings_.size(); ++candidate_model_i) {
    ID_TYPE current_node_size_threshold =
        candidate_model_settings_[candidate_model_i].gpu_ms_per_query / cpu_ms_per_series_;
    if (current_node_size_threshold < node_size_threshold_) {
      node_size_threshold_ = current_node_size_threshold;
    }
  }
  //QYL
  // void dstree::Allocator::set_training_data(
  //   const std::vector<dstree::Answers>& train_answers) {
  //     train_answers_ = train_answers;
  // }

#ifdef DEBUG
  spdlog::info("allocator node size threshold measured {:d}; default {:d}",
               node_size_threshold_, config_.get().filter_default_node_size_threshold_);
#endif

  if (node_size_threshold_ < config_.get().filter_default_node_size_threshold_) {
#ifdef DEBUG
    spdlog::info("allocator node size threshold measured {:d}; rectified to default {:d}",
                 node_size_threshold_, config_.get().filter_default_node_size_threshold_);
#endif

    node_size_threshold_ = config_.get().filter_default_node_size_threshold_;
  }
}

RESPONSE dstree::Allocator::push_filter_info(const FilterInfo &filter_info) {
  filter_infos_.push_back(filter_info);

  return SUCCESS;
}

struct TrialCache {
  TrialCache(dstree::Config &config,
             ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::vector<upcite::MODEL_SETTING> &candidate_model_settings,
             std::vector<dstree::FilterInfo> &filter_infos,
             ID_TYPE trial_nnode,
             ID_TYPE trial_nmodel,
             std::vector<ID_TYPE> &sampled_filter_idx,
             std::vector<VALUE_TYPE> &filter_pruning_ratios,
             ID_TYPE *trial_sample_i_ptr,
             pthread_mutex_t *sample_idx_mutex) :
      config_(config),
      thread_id_(thread_id),
      stream_(stream),
      candidate_model_settings_ref_(candidate_model_settings),
      filter_infos_ref_(filter_infos),
      trial_nnode_(trial_nnode),
      trial_nmodel_(trial_nmodel),
      sampled_filter_idx_ref_(sampled_filter_idx),
      filter_pruning_ratios_ref_(filter_pruning_ratios),
      trial_sample_i_ptr_(trial_sample_i_ptr),
      sample_idx_mutex_(sample_idx_mutex) {}
  ~TrialCache() = default;

  std::reference_wrapper<dstree::Config> config_;

  ID_TYPE thread_id_;

  at::cuda::CUDAStream stream_;

  std::reference_wrapper<std::vector<upcite::MODEL_SETTING>> candidate_model_settings_ref_;
  std::reference_wrapper<std::vector<dstree::FilterInfo>> filter_infos_ref_;

  ID_TYPE trial_nnode_;
  ID_TYPE trial_nmodel_;

  std::reference_wrapper<std::vector<ID_TYPE>> sampled_filter_idx_ref_;
  std::reference_wrapper<std::vector<VALUE_TYPE>> filter_pruning_ratios_ref_;

  ID_TYPE *trial_sample_i_ptr_;
  pthread_mutex_t *sample_idx_mutex_;
};

void trial_thread_F(TrialCache &trial_cache) {
  at::cuda::setCurrentCUDAStream(trial_cache.stream_);
  at::cuda::CUDAStreamGuard guard(trial_cache.stream_); // compiles with libtorch-gpu

#ifdef DEBUG
#ifndef DEBUGGED
  spdlog::debug("allocator thread {:d}", trial_cache.thread_id_);
  spdlog::debug("allocator candidate_model_settings_ref_.get().size() {:d}",
                trial_cache.candidate_model_settings_ref_.get().size());
  spdlog::debug("allocator filter_infos_ref_.get().size() {:d}", trial_cache.filter_infos_ref_.get().size());
  spdlog::debug("allocator trial_nnode_ {:d}", trial_cache.trial_nnode_);
  spdlog::debug("allocator trial_nmodel_ {:d}", trial_cache.trial_nmodel_);
  spdlog::debug("allocator sampled_filter_idx_ref_.get().size() {:d}",
                trial_cache.sampled_filter_idx_ref_.get().size());
  spdlog::debug("allocator filter_pruning_ratios_ref_.get().size() {:d}",
                trial_cache.filter_pruning_ratios_ref_.get().size());
  spdlog::debug("allocator trial_sample_i_ {:d}", *trial_cache.trial_sample_i_ptr_);
#endif
#endif

  while (true) {
    pthread_mutex_lock(trial_cache.sample_idx_mutex_);

#ifdef DEBUG
#ifndef DEBUGGED
    spdlog::debug("allocator thread {:d} locked, *trial_cache.trial_sample_i_ptr_ = {:d}",
                  trial_cache.thread_id_,
                  *trial_cache.trial_sample_i_ptr_);
#endif
#endif

    if ((*trial_cache.trial_sample_i_ptr_) >= trial_cache.trial_nnode_) {
      
      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      break;
    } else {
      // iterate over nodes (check all models for this node)
      // TODO iterate over sampled [node, model] pairs
      ID_TYPE trial_sample_i = *trial_cache.trial_sample_i_ptr_;
      *trial_cache.trial_sample_i_ptr_ = trial_sample_i + 1;

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} to unlock; trial_sample_i = {:d}, *trial_cache.trial_sample_i_ptr_ = {:d}",
                    trial_cache.thread_id_,
                    trial_sample_i,
                    *trial_cache.trial_sample_i_ptr_);
#endif
#endif

      pthread_mutex_unlock(trial_cache.sample_idx_mutex_);

      ID_TYPE filter_sample_pos = trial_cache.sampled_filter_idx_ref_.get()[trial_sample_i];

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} sampled_filter_id = {:d}",
                    trial_cache.thread_id_, filter_sample_pos);
#endif
#endif

      std::reference_wrapper<dstree::FilterInfo> filter_info = trial_cache.filter_infos_ref_.get()[filter_sample_pos];
      auto filter_ref = filter_info.get().node_.get().get_filter();

#ifdef DEBUG
#ifndef DEBUGGED
      spdlog::debug("allocator thread {:d} check node {:d}",
                    trial_cache.thread_id_,
                    filter_ref.get_id()
      );
#endif
#endif

      for (ID_TYPE model_i = 0; model_i < trial_cache.trial_nmodel_; ++model_i) {
        auto &candidate_model_setting = trial_cache.candidate_model_settings_ref_.get()[model_i];

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} check model {:s} on node {:d}",
                      trial_cache.thread_id_,
                      candidate_model_setting.model_setting_str,
                      filter_ref.get_id()
        );
#endif
#endif

        filter_ref.get().trigger_trial(candidate_model_setting);
        filter_ref.get().train(true);

        // 2-d array of [no. models, no. nodes]
        trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i] =
            filter_ref.get().get_val_pruning_ratio();

#ifdef DEBUG
#ifndef DEBUGGED
        spdlog::debug("allocator thread {:d} node {:d} model {:d} pruning ratio = {:.3f}",
                      trial_cache.thread_id_,
                      trial_sample_i,
                      model_i,
                      trial_cache.filter_pruning_ratios_ref_.get()[trial_cache.trial_nnode_ * model_i + trial_sample_i]
        );
#endif
#endif
      }
    }
  }
}




RESPONSE dstree::Allocator::trial_collect_mthread() {
  // printf("[DEBUG] Sorting filter_infos_ by node size (descending)\n");
  std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

  ID_TYPE end_i_exclusive = filter_infos_.size();
  // printf("[DEBUG] Initial end_i_exclusive: %ld\n", end_i_exclusive);

  while (end_i_exclusive > 1 && filter_infos_[end_i_exclusive - 1].node_.get().get_size()
      < config_.get().filter_default_node_size_threshold_) {
    end_i_exclusive -= 1;
  }
  // printf("[DEBUG] Adjusted end_i_exclusive: %ld\n", end_i_exclusive);
  ID_TYPE offset = 0;
  //----------------------filter_trial_nnode_=32(小数据集可能不适配), end_i_exclusive =4
  // end_i_exclusive表示通过筛选之后的叶子节点
  ID_TYPE step = end_i_exclusive / config_.get().filter_trial_nnode_;
  // printf("[DEBUG] Step size for sampling: %ld\n", step);

  auto sampled_filter_idx = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE sample_i = 0; sample_i < config_.get().filter_trial_nnode_; ++sample_i) {
    sampled_filter_idx.push_back(offset + sample_i * step);
    // printf("[DEBUG] Sampled filter index %ld: %ld\n", sample_i, offset + sample_i * step);
  }

  // 2-d array of [no. models, no. nodes]
  auto filter_pruning_ratios = upcite::make_reserved<VALUE_TYPE>(
      config_.get().filter_trial_nnode_ * candidate_model_settings_.size());
  // printf("[DEBUG] Initializing filter_pruning_ratios with size: %ld\n", filter_pruning_ratios.size());

  for (ID_TYPE i = 0; i < config_.get().filter_trial_nnode_ * candidate_model_settings_.size(); ++i) {
    filter_pruning_ratios.push_back(0);
  }

  std::vector<std::unique_ptr<TrialCache>> trial_caches;
  std::unique_ptr<pthread_mutex_t> sample_idx_mutex = std::make_unique<pthread_mutex_t>();
  ID_TYPE trial_sample_i = 0;

  // printf("[DEBUG] Initializing trial caches and threads\n");
  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    // spdlog::info("trial thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
    //              thread_id,
    //              static_cast<ID_TYPE>(new_stream.id()),
    //              static_cast<ID_TYPE>(new_stream.query()),
    //              static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu
    // printf("[DEBUG] Creating trial cache for thread %ld\n", thread_id);

    trial_caches.emplace_back(std::make_unique<TrialCache>(config_,
                                                           thread_id,
                                                           std::move(new_stream),
                                                           std::ref(candidate_model_settings_),
                                                           std::ref(filter_infos_),
                                                           config_.get().filter_trial_nnode_,
                                                           candidate_model_settings_.size(),
                                                           std::ref(sampled_filter_idx),
                                                           std::ref(filter_pruning_ratios),
                                                           &trial_sample_i,
                                                           sample_idx_mutex.get()));
  }

  std::vector<std::thread> threads;
  // printf("[DEBUG] Launching threads\n");

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    // printf("[DEBUG] Starting thread %ld\n", thread_id);
    threads.emplace_back(trial_thread_F, std::ref(*trial_caches[thread_id]));
  }

  // Print sizes for debugging
  // printf("[DEBUG] config_.get().filter_train_nthread_: %ld\n", config_.get().filter_train_nthread_);
  // printf("[DEBUG] trial_caches.size(): %ld\n", trial_caches.size());
  // Join threads
  // printf("[DEBUG] Joining threads\n");

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_train_nthread_; ++thread_id) {
    threads[thread_id].join();
  }

#ifdef DEBUG
  auto sampled_filter_ids = upcite::make_reserved<ID_TYPE>(config_.get().filter_trial_nnode_);
  for (ID_TYPE filter_i = 0; filter_i < config_.get().filter_trial_nnode_; ++filter_i) {
    sampled_filter_ids.push_back(filter_infos_[sampled_filter_idx[filter_i]].node_.get().get_id());
  }

  spdlog::info("allocator sampled node ids = {:s}",
    upcite::array2str(sampled_filter_ids.data(), sampled_filter_ids.size()));
  // printf("[DEBUG] Sampled node IDs: %s\n", upcite::array2str(sampled_filter_ids.data(), sampled_filter_ids.size()).c_str());

#endif

#ifdef DEBUG
  spdlog::info("allocator trial pruning ratios = {:s}",
               upcite::array2str(filter_pruning_ratios.data(), filter_pruning_ratios.size()));
  // printf("[DEBUG] Trial pruning ratios: %s\n", upcite::array2str(filter_pruning_ratios.data(), filter_pruning_ratios.size()).c_str());
#endif


  // printf("[DEBUG] Calculating pruning probabilities for each model\n");
  for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
    VALUE_TYPE mean = 0;
    for (ID_TYPE sample_i = 0; sample_i < sampled_filter_idx.size(); ++sample_i) {
      // 2-d array of [no. models, no. nodes]
      mean += filter_pruning_ratios[sampled_filter_idx.size() * model_i + sample_i];
    }
    candidate_model_settings_[model_i].pruning_prob = mean / sampled_filter_idx.size();

#ifdef DEBUG
    spdlog::info("allocator model {:s} pruning ratio = {:.3f}",
                 candidate_model_settings_[model_i].model_setting_str,
                 candidate_model_settings_[model_i].pruning_prob);
    // printf("[DEBUG] Model %s pruning ratio: %.3f\n",
    //               candidate_model_settings_[model_i].model_setting_str.c_str(),
    //               candidate_model_settings_[model_i].pruning_prob);
#endif
  }
  // printf("[DEBUG] Exiting trial_collect_mthread() function\n");
  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_cpu() {
  // test cpu_ms_per_series_
  auto batch_nbytes = static_cast<ID_TYPE>(
      sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().leaf_max_nseries_;
  auto trial_batch = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), batch_nbytes));

  auto distances = make_reserved<VALUE_TYPE>(config_.get().leaf_max_nseries_);

  if (config_.get().on_disk_) {
    // credit to https://stackoverflow.com/a/19728404
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<ID_TYPE> uni_i_d(0, config_.get().db_nseries_ - config_.get().leaf_max_nseries_);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      std::ifstream db_fin;
      db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);

      ID_TYPE batch_bytes_offset = static_cast<ID_TYPE>(
          sizeof(VALUE_TYPE)) * config_.get().series_length_ * uni_i_d(rng);

      db_fin.seekg(batch_bytes_offset);
      db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }

      db_fin.close();
      distances.clear();
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  } else {
    std::ifstream db_fin;
    db_fin.open(config_.get().db_filepath_, std::ios::in | std::ios::binary);
    db_fin.read(reinterpret_cast<char *>(trial_batch), batch_nbytes);

    auto start = std::chrono::high_resolution_clock::now();

    for (ID_TYPE trial_i = 0; trial_i < config_.get().allocator_cpu_trial_iterations_; ++trial_i) {
      distances.clear();

      for (ID_TYPE series_i = 0; series_i < config_.get().leaf_max_nseries_; ++series_i) {
        VALUE_TYPE distance = upcite::cal_EDsquare(trial_batch,
                                                   trial_batch + series_i * config_.get().series_length_,
                                                   config_.get().series_length_);
        distances.push_back(distance);
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    db_fin.close();

    cpu_ms_per_series_ = duration.count() / static_cast<double_t>(
        config_.get().allocator_cpu_trial_iterations_ * config_.get().leaf_max_nseries_);
  }

#ifdef DEBUG
  spdlog::info("allocator trial cpu time = {:.6f}mus", cpu_ms_per_series_);
#endif

  free(trial_batch);
  return SUCCESS;
}

RESPONSE dstree::Allocator::measure_gpu() {
  if (torch::cuda::is_available()) {
    bool measure_required = false;
    for (auto const &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        measure_required = true;
      }
    }

    if (!measure_required) {
      return SUCCESS;
    }

    std::random_device rd;
    std::mt19937 e2(rd());
    std::uniform_real_distribution<> dist(0, 1);
    ID_TYPE query_nbytes = sizeof(VALUE_TYPE) * config_.get().series_length_;
    auto random_input = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));

    for (ID_TYPE i = 0; i < config_.get().series_length_; ++i) {
      random_input[i] = dist(e2);
    }

    auto input_tsr_ = torch::from_blob(random_input,
                                       {1, config_.get().series_length_},
                                       torch::TensorOptions().dtype(TORCH_VALUE_TYPE));

    std::unique_ptr<torch::Device> device = nullptr;
    if (config_.get().filter_infer_is_gpu_) {
      device = std::make_unique<torch::Device>(torch::kCUDA,
                                               static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device = std::make_unique<torch::Device>(torch::kCPU);
    }
    input_tsr_ = input_tsr_.to(*device);

    auto trial_filter = std::make_unique<dstree::Filter>(config_, -1, input_tsr_);

    for (auto &model_setting_ref : candidate_model_settings_) {
      if (model_setting_ref.gpu_mem_mb <= constant::EPSILON) {
        // printf("Allocator::measure_gpu()调用collect_running_info \n");
        trial_filter->collect_running_info(model_setting_ref);
      }
    }

    free(random_input);
  }

  return SUCCESS;
}

RESPONSE dstree::Allocator::evaluate() {
  // test candidate_model_setting_.pruning_prob
  // printf("[DEBUG] Entering evaluate() function\n");
  // printf("[DEBUG] Calling trial_collect_mthread()\n");
  trial_collect_mthread();

  // calculate gain for [node_i, model_i]
  // printf("[DEBUG] Preparing filter_ids_, gains_matrix_, and mem_matrix_\n");
  // printf("[DEBUG] filter_infos_.size(): %ld\n", filter_infos_.size());
  // printf("[DEBUG] candidate_model_settings_.size(): %ld\n", candidate_model_settings_.size());

  filter_ids_.reserve(filter_infos_.size());
  gains_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());
  mem_matrix_.reserve(filter_infos_.size() * candidate_model_settings_.size());

  for (ID_TYPE filter_i = 0; filter_i < filter_infos_.size(); ++filter_i) {
    auto &filter_info = filter_infos_[filter_i];
    filter_info.score = 0;
    // printf("[DEBUG] Processing filter %ld, node ID: %ld\n", filter_i, filter_info.node_.get().get_id());

    filter_ids_.push_back(filter_info.node_.get().get_id());

    for (ID_TYPE model_i = 0; model_i < candidate_model_settings_.size(); ++model_i) {
      auto &candidate_model_setting_ = candidate_model_settings_[model_i];
      // printf("[DEBUG] Processing model %ld: %s\n", model_i, candidate_model_setting_.model_setting_str.c_str());

      // TODO support model in cpu
      double_t amortized_gpu_sps = static_cast<double_t>(candidate_model_setting_.gpu_ms_per_query)
          / static_cast<double_t>(filter_info.node_.get().get_size());

      // printf("[DEBUG] amortized_gpu_sps: %f, cpu_ms_per_series_: %f\n", amortized_gpu_sps, cpu_ms_per_series_);
    
      if (amortized_gpu_sps > cpu_ms_per_series_) {
        spdlog::error("allocator model {:s} slower than cpu: {:f} > {:f}",
                      candidate_model_setting_.model_setting_str, amortized_gpu_sps, cpu_ms_per_series_);
        // printf("[ERROR] Model %s is slower than CPU\n", candidate_model_setting_.model_setting_str.c_str());

      }

      auto gain = static_cast<VALUE_TYPE>(static_cast<double_t>(filter_info.node_.get().get_size())
          * static_cast<double_t>((1 - filter_info.external_pruning_probability_)
              * candidate_model_setting_.pruning_prob)
          * (cpu_ms_per_series_ - amortized_gpu_sps));
      // printf("[DEBUG] Calculated gain for filter %ld, model %ld: %f\n", filter_i, model_i, gain);

      if (gain < 0) {
        // forbid harmful plans
        // printf("[DEBUG] Gain is negative, setting gain to 0 and memory to %f\n", available_gpu_memory_mb_ + 1);

        gains_matrix_.push_back(0);
        mem_matrix_.push_back(available_gpu_memory_mb_ + 1);
      } else {
        // printf("[DEBUG] Gain is positive, adding gain: %f and memory: %f\n", gain, candidate_model_setting_.gpu_mem_mb);

        gains_matrix_.push_back(gain);
        mem_matrix_.push_back(candidate_model_setting_.gpu_mem_mb);
      }

      if (gain > filter_info.score) {
        // printf("[DEBUG] Updating filter %ld score to %f\n", filter_i, gain);

        filter_info.score = gain;
        filter_info.model_setting = candidate_model_setting_;
      }
    }
  }
  // printf("[DEBUG] Exiting evaluate() function\n");

  return SUCCESS;
}






//这个函数是选择合适的叶子节点插入filter，具体选择是通过gain方法，bi = (1 − p^lb)× (p^F × t^S ×|Ni|− t^F)
// bi > 0 ⇒ |Ni| > a*(t^F/t^S)
RESPONSE dstree::Allocator::assign() {
  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;
  // printf("[DEBUG] Entering assign() function\n");

  // printf("[DEBUG] config_.get().filter_allocate_is_gain_ = %d\n", config_.get().filter_allocate_is_gain_);

  if (config_.get().filter_allocate_is_gain_) {
    //-------------2. 增益优先分配策略 (filter_allocate_is_gain_ = true)
    // printf("[DEBUG] Using gain-based allocation\n");

    evaluate();

    // -------------情况 2.1: 仅一个候选模型配置 
    if (candidate_model_settings_.size() == 1) {
      // printf("[DEBUG] Sorting filter_infos_ by score (descending)\n");
      // 按增益分数降序排序
      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterScore);
      
      // --------2.1.1：处理无有效增益的情况，此时回退到基于节点大小的分配
      if (filter_infos_[0].score <= 0) {
        spdlog::error("allocator gain-based allocation failed; revert to size-based allocation");
        // printf("[DEBUG] Gain-based allocation failed, reverting to size-based allocation\n");
        for (auto &filter_info : filter_infos_) {
           //检查节点大小和内存限制
          if ((filter_info.node_.get().get_size() > config_.get().filter_node_size_threshold_
              || config_.get().to_profile_filters_)
              && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
            //激活过滤器    
            // printf("[DEBUG] Attempting to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());
            if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
              allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
              allocated_filters_count += 1;
              // printf("[DEBUG] Successfully activated filter for node ID: %ld\n", filter_info.node_.get().get_id());
            }
          } else {
            // printf("[DEBUG] Skipping filter for node ID: %ld (size or memory constraint)\n", filter_info.node_.get().get_id());
            break; // 内存不足时终止循环
          }
        }
      } else {
        //-------------2.1.2: 正常增益分配流程
        // printf("[DEBUG] Proceeding with gain-based allocation\n");

        for (auto &filter_info : filter_infos_) {
          if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_) {
            break;
          }
          // 根据配置或增益分数激活
          if (config_.get().to_profile_filters_ || filter_info.score > 0) {
            // printf("[DEBUG] Attempting to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());

            if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
              allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
              allocated_filters_count += 1;
              // printf("[DEBUG] Successfully activated filter for node ID: %ld\n", filter_info.node_.get().get_id());

            }
          } else if (filter_info.score <= 0) {
            // printf("[DEBUG] Failed to activate filter for node ID: %ld\n", filter_info.node_.get().get_id());
            break; // 增益不足时终止
          }
        }
      }
    } 

  } else {
    // default: implant the default model to all leaf nodes
    printf("[DEBUG] Using size-based allocation\n");

    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

    if (config_.get().filter_model_setting_str_.empty() || candidate_model_settings_.empty()) {
      spdlog::error("allocator default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      ID_TYPE filter_node_size_threshold = config_.get().filter_fixed_node_size_threshold_;
      if (filter_node_size_threshold < 0) {
        filter_node_size_threshold = node_size_threshold_;
      }
      spdlog::info("allocator assign filter_node_size_threshold {:d}, measured {:d} fixed {:d}",
                   filter_node_size_threshold, node_size_threshold_, config_.get().filter_fixed_node_size_threshold_);

      for (auto &filter_info : filter_infos_) {
        if ((filter_info.node_.get().get_size() >= filter_node_size_threshold
            || config_.get().to_profile_filters_)
            && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        } else {
          filter_info.node_.get().deactivate_filter();
        }
      }

    }
  }

  spdlog::info("allocator assigned {:d} models of {:.3f}mb/{:.3f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}




RESPONSE dstree::Allocator::reassign() {
  if (candidate_model_settings_.size() != 1) {
    spdlog::error("allocator reallocation only supports single candidate");
    return FAILURE;
  }

  VALUE_TYPE allocated_gpu_memory_mb = 0;
  ID_TYPE allocated_filters_count = 0;

  if (config_.get().filter_allocate_is_gain_) {
    if (candidate_model_settings_.size() == 1) {
      for (auto &filter_info : filter_infos_) {
        filter_info.model_setting = candidate_model_settings_[0];
      }

      auto min_nseries = static_cast<ID_TYPE>(candidate_model_settings_[0].gpu_ms_per_query / cpu_ms_per_series_);
      spdlog::info("allocator re-assign (single), derived min_nseries = {:d}", min_nseries);

      std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

      for (auto &filter_info : filter_infos_) {
        if (allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb > available_gpu_memory_mb_
            || filter_info.node_.get().get_size() < min_nseries) {
          break;
        } else {
          assert(filter_info.model_setting.get().gpu_mem_mb > 0);
          assert(filter_info.model_setting.get().gpu_ms_per_query > 0);
        }

        if (filter_info.node_.get().activate_filter(filter_info.model_setting) == SUCCESS) {
          allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
          allocated_filters_count += 1;
        }
      }
    } else {
      // TODO is reassignment possible for multi models?
    }
  } else {
    // default: implant the default model to all leaf nodes
    std::sort(filter_infos_.begin(), filter_infos_.end(), dstree::compDecreFilterNSeries);

    if (config_.get().filter_model_setting_str_.empty() || candidate_model_settings_.empty()) {
      spdlog::error("allocator default model setting does not exist (set by --filter_model_setting=)");
      return FAILURE;
    } else {
      if (candidate_model_settings_.size() > 1) {
        spdlog::warn("allocator > 1 default model settings found; use the first {:s}",
                     candidate_model_settings_[0].model_setting_str);
      }

      ID_TYPE filter_node_size_threshold = config_.get().filter_fixed_node_size_threshold_;
      if (filter_node_size_threshold < 0) {
        filter_node_size_threshold = node_size_threshold_;
      }
      spdlog::info("allocator reassign filter_node_size_threshold {:d}, measured {:d} fixed {:d}",
                   filter_node_size_threshold, node_size_threshold_, config_.get().filter_fixed_node_size_threshold_);

      for (auto &filter_info : filter_infos_) {
//        spdlog::debug("allocator reassign node_.get_size {:d}, has_trained_filter {:d}",
//                      filter_info.node_.get().get_size(),
//                      filter_info.node_.get().has_trained_filter());

        if (filter_info.node_.get().get_size() >= filter_node_size_threshold
            && filter_info.node_.get().has_trained_filter()
            && allocated_gpu_memory_mb + filter_info.model_setting.get().gpu_mem_mb <= available_gpu_memory_mb_) {
          if (filter_info.node_.get().activate_filter(candidate_model_settings_[0]) == SUCCESS) {
            allocated_gpu_memory_mb += filter_info.model_setting.get().gpu_mem_mb;
            allocated_filters_count += 1;
          }
        } else {
          filter_info.node_.get().deactivate_filter();
        }
      }
    }
  }

  spdlog::info("allocator reassigned {:d} models of {:.1f}/{:.1f}mb gpu memory",
               allocated_filters_count, allocated_gpu_memory_mb, available_gpu_memory_mb_);
  return SUCCESS;
}



/*
Revised by Yanlin Qi
1. add KNN search process to compute recall
2. add multiple batches 
*/ 
RESPONSE dstree::Allocator::set_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes){
  printf("allocator filter_infos_.size() = %d\n", filter_infos_.size());

  // 创建节点ID到filter_infos_索引的映射
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }

  // 打印映射表（调试用）
  printf("节点ID到索引映射表：\n");
  for (const auto& [node_id, index] : node_id_to_index) {
    printf("  节点ID %d -> 索引 %zu\n", node_id, index);
  }

  if (!is_recall_calculated_) { //初始为false，执行内部代码（计算Conformal校准所需的示例数量）
    ID_TYPE num_conformal_examples, num_train_examples;

    if (config_.get().filter_train_num_global_example_ > 0) {
      num_train_examples = static_cast<ID_TYPE>(config_.get().filter_train_num_global_example_ * config_.get().filter_train_val_split_);
      ID_TYPE num_global_valid_examples = config_.get().filter_train_num_global_example_ - num_train_examples;
      num_conformal_examples = num_global_valid_examples;

    } else { // 仅全局数据的场景 only contains global examples
      num_train_examples = static_cast<ID_TYPE>(config_.get().filter_train_nexample_ * config_.get().filter_train_val_split_);
      ID_TYPE num_valid_examples = config_.get().filter_train_nexample_ - num_train_examples;
      num_conformal_examples = num_valid_examples;  
    }

    // 2.2 初始化存储最近邻距离和对应过滤器ID的数组
    const ID_TYPE K = config_.get().n_nearest_neighbor_; 
    for (ID_TYPE sorted_error_i = 0; sorted_error_i < num_conformal_examples + 2; ++sorted_error_i) {
      
      ID_TYPE total_hit_count = 0; //存储所有query的找到的真实KNN数量
      const ID_TYPE batch_total_knn = K * num_conformal_examples;

      printf("\n----- sorted_error_i = %d -----\n", sorted_error_i);  // 打印当前误差区间索引

      for (ID_TYPE query_i = 0; query_i < num_conformal_examples; ++query_i) {
        // 原始index.cc的filter_collect()函数中，收集的global_1nn_distances,global_bsf_distances,都是包含了训练集和校准集全部的query
        // 这里需要区分训练集和校准集，训练集的query_id从0到num_train_examples-1, 校准集的query_id从num_train_examples到num_train_examples+num_conformal_examples-1
        ID_TYPE current_query_id = num_train_examples + query_i;

        printf("\n[Query %d] current_query_id = %d\n", query_i, current_query_id);
        
        // 获取当前查询的节点分布
        auto it = query_knn_nodes.find(current_query_id);
        if (it == query_knn_nodes.end()) {
          fprintf(stderr, "Query %d not found\n", current_query_id);
          continue;
        }       
        // node_counts 是 std::unordered_map<ID_TYPE, ID_TYPE>, 表示 <节点ID, 该节点下的真实KNN数量>
        const auto& node_counts = it->second;  
        ID_TYPE hit_count = 0;
        ID_TYPE wrong_pruned_count = 0; // 统计错误减枝的节点数
        ID_TYPE no_filter_count = 0;    // 统计没有使用过滤器的节点数
        
        for (const auto& [node_id, count_in_node] : node_counts) {
            printf("[node_id %d] KNN series = %d\n", node_id, count_in_node);
            
            // 使用映射查找节点，而不是直接用node_id作为索引
            auto map_it = node_id_to_index.find(node_id);
            if (map_it == node_id_to_index.end()) {
                printf("警告：节点ID %d 未找到对应的索引\n", node_id);
                continue;  // 跳过这个节点
            }
            
            // 找到节点，使用正确的索引访问
            size_t node_index = map_it->second;
            if (node_index >= filter_infos_.size()) {
                printf("错误：节点索引 %zu 超出 filter_infos_ 范围 (size=%zu)\n", 
                       node_index, filter_infos_.size());
                continue;
            }
            
            auto& filter_info = filter_infos_[node_index];
            auto& target_node = filter_info.node_;

            if (target_node.get().has_active_filter()) {
              printf("======target_node has_active_filter ========\n"); 
              VALUE_TYPE abs_error = target_node.get().get_filter_abs_error_interval_by_pos(sorted_error_i);
              VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
              VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
              // printf("abs_error: %.3f, pred_distance: %.3f, bsf_distance: %.3f\n", abs_error, pred_distance, bsf_distance);
              // 判断是否覆盖该节点所有KNN
              if (pred_distance - abs_error <= bsf_distance) {
                hit_count += count_in_node; // 累加该节点的贡献量
                printf("COVERED (hit += %d)", count_in_node);
              } 

            } else {
              printf("---------- target_node not active_filter ----------- \n"); 
              hit_count += count_in_node; // 无过滤器则全部命中
              printf("no_filter -> fully COVERED (hit += %d)", count_in_node);
            }
            printf("\n");
          }
          
        total_hit_count += hit_count; // 累加所有query的hit_count
        printf("[Query %d] hit_count = %d\n", query_i, hit_count);
      } 
        // end query loop
        // [10] 打印最终统计结果
      printf("\n===== sorted_error_i = %d =====\n", sorted_error_i);
      printf("Total hit count: %d\n", total_hit_count);
      printf("Batch total knn: %d\n", batch_total_knn);
      printf("avg Recall: %.2f%%\n", (total_hit_count * 100.0) / batch_total_knn);
      
      // 存储当前error分位数对应的平均召回率
      ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
      if (!is_recall_calculated_) {
        // 首次计算时，存储召回率
        validation_recalls_.push_back(recall);
        
        // 更新最小验证召回率
        if (min_validation_recall_ > recall) {
          min_validation_recall_ = recall;
        }
      }
    }
    
    if (!is_recall_calculated_) {
      // 确保validation_recalls_数组的值单调递减
      if (validation_recalls_.size() > 0) {
        validation_recalls_[validation_recalls_.size() - 1] = 1 - constant::EPSILON_GAP;
        printf("constant::EPSILON_GAP = %.10f\n", constant::EPSILON_GAP);
        for (ID_TYPE backtrace_i = validation_recalls_.size() - 2; backtrace_i >= 0; --backtrace_i) {
          if (validation_recalls_[backtrace_i] > validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP) {
            validation_recalls_[backtrace_i] = validation_recalls_[backtrace_i + 1] - constant::EPSILON_GAP;
          }
        }
      }
      
      // 打印修改后的validation_recalls_数组
      for (size_t i = 0; i < validation_recalls_.size(); ++i) {
        printf("validation_recalls_[%zu] = %.5f\n", i, validation_recalls_[i]);
      }
      
      is_recall_calculated_ = true;
      
      // 如果使用样条连续方法，为每个filter拟合recall到误差分位数的映射
      if (config_.get().filter_conformal_is_smoothen_) {
        printf("filter_infos_.size() = %ld\n", static_cast<long>(filter_infos_.size()));
        for (auto &filter_info : filter_infos_) {
          printf("Processing node with ID: %ld\n", static_cast<long>(filter_info.node_.get().get_id()));
          if (filter_info.node_.get().has_active_filter()) {
            filter_info.node_.get().fit_filter_conformal_spline(validation_recalls_);
        }
      }
      }
    }
  }
  
  // 根据用户指定的召回率阈值，动态调整误差分位数
  printf("---------------此时根据召回率阈值利用样条插值和离散方法来动态调整误差分位数----------------\n");
  printf("min_validation_recall_ = %.5f, filter_conformal_recall_ = %.5f\n", 
         min_validation_recall_, config_.get().filter_conformal_recall_);
  
  // 1. 如果最小验证召回率已经满足用户要求，不需要额外调整
  if (min_validation_recall_ > config_.get().filter_conformal_recall_) {
    printf("-------当前校准集的最小avgRecall都已经大于用户定义的filter_conformal_recall_, 此时不需要额外调整-------\n");
    
    spdlog::info("allocator requested recall {:.3f} out of trained min {:.3f}; do NOT adjust",
                 config_.get().filter_conformal_recall_, min_validation_recall_);
    // 一个校准集会产生一个filter_infos_
    for (auto &filter_info : filter_infos_) {
      if (filter_info.node_.get().has_active_filter()) {
        if (filter_info.node_.get().set_filter_abs_error_interval(0) == FAILURE) {
          spdlog::error("allocator failed to get node {:d} conformed by 0",
                        filter_info.node_.get().get_id());
        }
        spdlog::info("allocator node {:d} abs_error {:.3f} at min {:.3f}; requested {:.3f}",
                     filter_info.node_.get().get_id(),
                     filter_info.node_.get().get_filter_abs_error_interval(),
                     min_validation_recall_, config_.get().filter_conformal_recall_);
      }
    }


  } else { 
    printf("--------2. 这里表示当前并不是所有的分位数下的avgrecall都能满足用户阈值, 那么此时需要计算合适的误差分位数-----\n");
    // 2.1 使用样条函数预测用户要求的recall的误差分位数
    if (config_.get().filter_conformal_is_smoothen_) {
      printf("------2.1: using spline regression to compute oi(delta) when given recall-------\n");
      // 定义计数器
      int total_filters = 0;      // 总过滤器数量
      int active_filters = 0;     // 激活的过滤器数量
      //一个校准集会产生一个filter_infos_，这里需要遍历filter_infos_来计算每个filter的oi(delta)
      for (auto &filter_info : filter_infos_) {
        total_filters++; // 每次循环递增总过滤器数量
        
        if (filter_info.node_.get().has_active_filter()) {
          active_filters++; // 如果过滤器激活，递增激活过滤器数量
          printf("filter node ID: %ld\n", static_cast<long>(filter_info.node_.get().get_id()));
          // 根据用户给定的recall阈值，通过插值计算对应的alpha值
          if (filter_info.node_.get().set_filter_abs_error_interval_by_recall(config_.get().filter_conformal_recall_) == FAILURE) {
            spdlog::error("allocator failed to get node {:d} conformed at recall {:.3f}",
                          filter_info.node_.get().get_id(), config_.get().filter_conformal_recall_);
            printf("Failed to set filter abs error interval for node with ID: %ld\n", 
                   static_cast<long>(filter_info.node_.get().get_id()));
          }
        }
      }
      // 打印总过滤器数量和激活的过滤器数量
      printf("\n !!!!!!!!!!!!!!!!!!  Total filters: %d, Active filters: %d\n", total_filters, active_filters);


    } else {
      printf("------2.2: using discrete to compute oi(delta) when given recall-------\n");
      // 使用离散方法
      ID_TYPE last_recall_i = validation_recalls_.size() - 1;
      // 倒序访问validation_recalls_中的recall，找到满足用户要求的最小分位数
      for (ID_TYPE recall_i = validation_recalls_.size() - 2; recall_i >= 0; --recall_i) {
        if (validation_recalls_[recall_i] < config_.get().filter_conformal_recall_) {
          last_recall_i = recall_i + 1;
          printf("allocator reached recall %.5f with error_i %.5f (%d/%zu, 2 sentries included)\n",
                 validation_recalls_[last_recall_i],
                 static_cast<VALUE_TYPE>(last_recall_i) / validation_recalls_.size(),
                 last_recall_i,
                 validation_recalls_.size());
          break;
        }
      }
      // 根据找到的分位数设置每个过滤器的误差区间
      for (auto &filter_info : filter_infos_) {
        if (filter_info.node_.get().has_active_filter()) {
          if (filter_info.node_.get().set_filter_abs_error_interval_by_pos(last_recall_i) == FAILURE) {
            printf("allocator failed to get node %ld conformed with abs %.5f at pos %ld\n",
                   filter_info.node_.get().get_id(),
                   filter_info.node_.get().get_filter_abs_error_interval_by_pos(last_recall_i),
                   last_recall_i);
          }
        }
      }
    }
  }
  return SUCCESS;
}




// 记录校准集所有批次的query的真实knn节点对应的真实距离，minbsf，pred_distance，abs_error，true_error
RESPONSE dstree::Allocator::document_cp_dist(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
  // 1. 创建节点ID到filter_infos_在数组中索引位置的映射（方便查找）
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }

  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  // 2. 查询校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  
  bool found_calibration_info = false;
  printf("Allocator filter_infos_.size() = %zu\n", filter_infos_.size());

  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {
    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();

      if (!filter.get_batch_calib_query_ids().empty()) {
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size(); // 假设所有批次大小相似
        found_calibration_info = true;
        printf("从节点 %ld 获取到校准批次信息: %ld 批次, 每批约 %ld 样本\n", 
               filter_info.node_.get().get_id(), num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到任何有效的校准批次信息\n");
    return FAILURE;
  }
  
  // 3. 确定误差分位数的数量
  ID_TYPE num_error_quantiles = examples_per_batch + 2; // 加2是为了添加哨兵值
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  
  // 4. 创建CSV文件用于保存数据
  std::string save_path = config_.get().save_path_;
  // 如果save_path不为空，则创建目录
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      fs::create_directories(save_path);
    }
  }
  // 创建CSV文件
  std::string csv_path = save_path + "/calibration_data.csv";
  printf("当前CSV保存路径: %s\n", csv_path.c_str());
  std::ofstream csv_file(csv_path);
  if (!csv_file.is_open()) {
    printf("错误: 无法创建CSV文件 %s\n", csv_path.c_str());
    return FAILURE;
  }
  
  // 写入CSV表头
  csv_file << "batch_id,query_id,node_id,true_distance,pred_distance,bsf_distance,true_error,abs_error\n";
  
  // 只使用最后一个误差分位数
  ID_TYPE error_i = num_error_quantiles - 1;
  printf("使用最后一个误差分位数: %ld\n", error_i);

  int total_rows = 0;
  
  // 5. 中层循环: 遍历校准集批次 (限制为100个批次)
  ID_TYPE max_batches = std::min(num_batches, static_cast<ID_TYPE>(100));
  for (ID_TYPE batch_i = 0; batch_i < max_batches; ++batch_i) {
    const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
    ID_TYPE batch_size = current_batch_query_ids.size();
    
    // 6. 内层循环: 遍历当前批次的每个查询
    for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
      ID_TYPE current_query_id = current_batch_query_ids[query_idx];
      
      // 获取当前查询的节点分布
      auto it = query_knn_nodes.find(current_query_id);
      if (it == query_knn_nodes.end()) {
        // 如果找不到查询，输出警告并继续
        fprintf(stderr, "查询 %d 未找到\n", current_query_id);
        continue;
      }
      
      // node_counts: <节点ID, 该节点下的真实KNN数量>
      const auto& node_counts = it->second;
      
      // 遍历当前查询的所有相关节点
      for (const auto& [node_id, count_in_node] : node_counts) {
        // 使用映射查找节点
        auto map_it = node_id_to_index.find(node_id);
        if (map_it == node_id_to_index.end()) {
          printf("警告:节点ID %ld 未找到对应的索引\n", node_id);
          continue;
        }
        
        // 找到节点，使用正确的索引访问
        size_t node_index = map_it->second;
        if (node_index >= filter_infos_.size()) {
          printf("错误：节点索引 %zu 超出 filter_infos_ 范围 (size=%zu)\n", 
                 node_index, filter_infos_.size());
          continue;
        }
        
        auto& filter_info = filter_infos_[node_index];
        auto& target_node = filter_info.node_;
        
        if (target_node.get().has_active_filter()) {
          // 获取当前批次、当前误差分位数下的对应filter的误差
          VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
          VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
          VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
          VALUE_TYPE true_distance = target_node.get().get_filter_nn_distance(current_query_id);
          VALUE_TYPE true_error = pred_distance - true_distance;
          
          // 所有的误差，包括剪枝和没剪枝的
          // 写入CSV文件
          csv_file << batch_i << "," 
                   << current_query_id << "," 
                   << node_id << "," 
                   << true_distance << "," 
                   << pred_distance << "," 
                   << bsf_distance << "," 
                   << true_error << "," 
                   << abs_error << "\n";
                   
          total_rows++;
        }
      }
    }
    // 每处理完10个批次打印一次进度
    if (batch_i % 10 == 0 || batch_i == max_batches - 1) {
      printf("已处理 %ld/%ld 批次, 当前已写入 %d 行数据\n", batch_i + 1, max_batches, total_rows);
    }
  }
  
  csv_file.close();
  printf("数据已保存到 %s, 共 %d 行\n", csv_path.c_str(), total_rows);
  
  // 安全释放临时变量，避免析构时的问题
  node_id_to_index.clear();
  batch_query_ids.clear();
  
  return SUCCESS;
}






// 实现多批次校准集的置信区间计算
RESPONSE dstree::Allocator::set_batch_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes) {
  printf("\n ------Allocator::set_batch_confidence_from_recall ------ \n");
  // 最终设置的满足覆盖率和召回率的min_satisfying_error_i为-1表示未找到
  ID_TYPE max_satisfying_error_i = -1;  // 初始化为-1表示未找到

  // 1. 创建节点ID到filter_infos_在数组中索引位置的映射（方便查找）
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }

  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  
  // 3. 查询校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  bool found_calibration_info = false;
  printf("Allocator filter_infos_.size() = %zu\n", filter_infos_.size());

  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {

    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();

      if (!filter.get_batch_calib_query_ids().empty()) {
        //在训练过程中，每个过滤器都会调用generate_calibration_batches，生成并存储batch_calib_query_ids_。
        //这个检查确保找到的过滤器已经完成了这一步骤。
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size(); // 假设所有批次大小相似
        found_calibration_info = true;
        printf("从节点 %ld 获取到校准批次信息: %ld 批次, 每批约 %ld 样本\n", filter_info.node_.get().get_id(), num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到任何有效的校准批次信息\n");
    return FAILURE;
  }
  
  // 4. 确定误差分位数的数量
  ID_TYPE num_error_quantiles = examples_per_batch + 2; // 加2是为了添加哨兵值
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  // 5. 存储每个误差分位数下满足召回率要求的批次数量
  std::vector<ID_TYPE> satisfying_batches(num_error_quantiles, 0);
  if (!is_recall_calculated_) { // 初始为false，执行内部代码
    // 初始化多批次校准集的召回率存储结构
    batch_validation_recalls_.clear();
    batch_validation_recalls_.resize(num_batches);
    
    // 6. 外层循环: 遍历误差分位数
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      // 7. 中层循环: 遍历校准集批次
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        // 获取当前batch i的所有query id
        const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
        ID_TYPE batch_size = current_batch_query_ids.size();
        
        ID_TYPE total_hit_count = 0; // 存储当前批次所有query找到的真实KNN数量
        const ID_TYPE batch_total_knn = K * batch_size;
        
        // 8. 内层循环: 遍历当前批次的每个查询
        for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
          // 获取当前query的id
          ID_TYPE current_query_id = current_batch_query_ids[query_idx]-1;
          auto it = query_knn_nodes.find(current_query_id);
          if (it == query_knn_nodes.end()) {
            // 如果找不到查询，输出警告并继续
            fprintf(stderr, "查询 %d 未找到\n", current_query_id);
            continue;
          }
          // node_counts: <节点ID, 该节点下的真实KNN数量>
          const auto& node_counts = it->second;
          ID_TYPE hit_count = 0;
          ID_TYPE wrong_pruned_count = 0; // 统计错误减枝的节点数
          ID_TYPE no_filter_count = 0;    // 统计没有使用过滤器的节点数
          // 遍历当前查询的所有相关节点
          for (const auto& [node_id, count_in_node] : node_counts) {
            // 使用映射查找节点
            // map_it: <节点ID, 该节点在filter_infos_中的索引位置>
            auto map_it = node_id_to_index.find(node_id);
            if (map_it == node_id_to_index.end()) {
              printf("警告:节点ID %ld 未找到对应的索引\n", node_id);
              continue;
            }
            // 找到节点，使用正确的索引
            size_t node_index = map_it->second;
            if (node_index >= filter_infos_.size()) {
              printf("错误：节点索引 %zu 超出 filter_infos_ 范围 (size=%zu)\n", 
                     node_index, filter_infos_.size());
              continue;
            }
            auto& filter_info = filter_infos_[node_index];
            auto& target_node = filter_info.node_;

            // spdlog::info("\n========= 在filter_allocator.cc中打印filter {}的距离向量 ========", target_node.get().get_id());
            // spdlog::info("query_id={} node id={}", current_query_id, target_node.get().get_id());
            // target_node.get().get_filter().get().debug_print_global_vectors();

            if (target_node.get().has_active_filter()) {
              // spdlog::debug("set_confidence_from_recall: current_query_id={}, node: {}", current_query_id, target_node.get().get_id());
              // 获取当前批次、当前误差分位数下的对应filter的误差，所以这里需要传入batch_i和error_i，但是不需要传入current_query_id，
              // 因为要计算当前batch下的所有query在同一个误差分位数下的recall。
              VALUE_TYPE abs_error = target_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
              VALUE_TYPE bsf_distance = target_node.get().get_filter_bsf_distance(current_query_id);
              VALUE_TYPE pred_distance = target_node.get().get_filter_pred_distance(current_query_id);
              VALUE_TYPE true_distance = target_node.get().get_filter_nn_distance(current_query_id);
              VALUE_TYPE true_error = pred_distance - true_distance;
              // spdlog::info("========= 统计所有query到knn节点的距离误差========= ");
              // spdlog::info("query_id={} node id={}  minbsf={:.3f}, true_dist={:.3f}, pred_dist={:.3f}, true_error={:.3f} error_quantile={:.3f}, batch_i={}", 
              //             current_query_id, target_node.get().get_id(), bsf_distance, true_distance, pred_distance, true_error, abs_error, batch_i);
              // spdlog::info("----------开始剪枝,收集所有分位数下的recall----------");
              // 判断是否覆盖该节点所有KNN
              if (pred_distance - abs_error <= bsf_distance) {
                // spdlog::info("pred_dist={:.3f} - abs_error={:.3f} <=  minbsf={:.3f}", pred_distance, abs_error, bsf_distance);
                // spdlog::debug("pred_distance: {:.3f}, abs_error: {:.3f}, bsf_distance: {:.3f}", pred_distance, abs_error, bsf_distance);
                hit_count += count_in_node; // 累加该节点的贡献量
                
              } else {
                // spdlog::info("pred_dist={:.3f} - abs_error={:.3f} >  minbsf={:.3f}", pred_distance, abs_error, bsf_distance);
                // 当前query在当前batch_i下，当前error_i下被错误减枝 
                wrong_pruned_count += 1; // 统计错误减枝的次数
              }

            } else {
              hit_count += count_in_node; // 无过滤器则全部命中
              // 当前query在当前batch_i下，当前error_i下没有使用filter，完全访问真实knn_nodes，所以全部命中
              no_filter_count += 1; // 统计没有使用过滤器的次数
            }
          }
          total_hit_count += hit_count; // 累加当前batch下的所有query的hit_count

        }
        // 内层循环结束 (查询遍历完成)
        // 计算当前批次、当前误差分位数下的平均召回率
        ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
        batch_validation_recalls_[batch_i].push_back(recall);
      }
      // 中层循环结束 (批次遍历完成)
    }
  
    // 外层循环结束 (误差分位数遍历完成)
    // 计算和应用(recall, coverage)对
    if (calculate_recall_coverage_pairs() != SUCCESS) {
      printf("警告: 计算(recall, coverage)对失败\n");
    }
    is_recall_calculated_ = true;
  }
  // printf("处理完毕，准备安全清理资源\n");
  // 安全释放临时变量，避免析构时的问题
  node_id_to_index.clear();
  batch_query_ids.clear();
  satisfying_batches.clear();
  // printf("set_batch_confidence_from_recall函数执行完成，准备返回\n");
  return SUCCESS;  // 添加明确的返回语句
}




// 模拟完整dstree搜索过程，重新计算准确的recall
RESPONSE dstree::Allocator::simulate_full_search_for_recall(std::shared_ptr<dstree::Node> root) {
  printf("\n ------Allocator::simulate_full_search_for_recall ------ \n");
  
  // 1. 从Node ID快速找到对应的Filter index， Filter Index = 该Filter在filter_infos_数组中的位置， Filter ID = Node ID（它们是同一个值）
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }
  printf("总共有 %zu 个filter\n", filter_infos_.size());
  spdlog::info("总共有 {} 个filter", filter_infos_.size());
  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  
  // 2. 获取校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  
  bool found_calibration_info = false;
  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {
    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      if (!filter.get_batch_calib_query_ids().empty()) {
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size();
        found_calibration_info = true;
        printf("获取校准批次信息: %ld 批次, 每批 %ld 样本\n", num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到校准批次信息\n");
    return FAILURE;
  }
  
  ID_TYPE num_error_quantiles = examples_per_batch + 2;
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  
  // 添加调试信息：打印关键配置参数
  printf("关键配置参数:\n");
  printf("  K (最近邻数量): %ld\n", K);
  printf("  批次数量: %ld\n", num_batches);
  printf("  每批样本数: %ld\n", examples_per_batch);
  printf("  叶子节点总数: %zu\n", filter_infos_.size());
  printf("  每批总KNN数 (K * batch_size): %ld\n", K * examples_per_batch);

  if (!is_recall_calculated_) { // 初始为false，执行内部代码

    // 3. 重新初始化召回率存储结构
    batch_validation_recalls_simulated_.clear();
    batch_validation_recalls_simulated_.resize(num_batches);
    
    // 4. 外层循环: 遍历误差分位数
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      printf("处理误差分位数 %ld/%ld\n", error_i + 1, num_error_quantiles);
      
      // 5. 中层循环: 遍历校准集批次
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
        ID_TYPE batch_size = current_batch_query_ids.size();
        ID_TYPE total_hit_count = 0;
        const ID_TYPE batch_total_knn = K * batch_size;
        
        // 6. 内层循环: 遍历当前批次的每个查询
        for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
          ID_TYPE current_query_id = current_batch_query_ids[query_idx] - 1;
          
          // 7. 模拟完整的dstree搜索过程
          ID_TYPE hit_count = simulate_dstree_search_for_query(
              current_query_id, batch_i, error_i, node_id_to_index, root);
          
          total_hit_count += hit_count;
        }
        
        // 计算当前批次、当前误差分位数下的召回率
        ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
        batch_validation_recalls_simulated_[batch_i].push_back(recall);
        
        // 添加调试信息：打印详细的计算过程
        // printf("  批次 %ld/%ld, 误差分位数 %ld/%ld: total_hit_count=%ld, batch_total_knn=%ld, 召回率=%.4f\n", 
        //        batch_i + 1, num_batches, error_i + 1, num_error_quantiles, 
        //        total_hit_count, batch_total_knn, recall);
        
        // 检查召回率是否超出合理范围
        if (recall > 1.0) {
          printf("警告: 召回率 %.4f 超过1.0，可能存在计算错误!\n", recall);
          printf("       total_hit_count=%ld, batch_total_knn=%ld\n", total_hit_count, batch_total_knn);
        }
        
        // 打印进度信息
        if (batch_i % 10 == 0) {
          printf("  批次 %ld/%ld, 召回率: %.4f\n", batch_i + 1, num_batches, recall);
        }
      }
    }

    if (calculate_recall_coverage_pairs() != SUCCESS) {
        printf("警告: 计算(recall, coverage)对失败\n");
    }
    is_recall_calculated_ = true;
    
  }
    // 安全释放临时变量，避免析构时的问题
  node_id_to_index.clear();
  batch_query_ids.clear();

  // printf("完整搜索模拟完成\n");
  return SUCCESS;
}



// 模拟单个查询的dstree搜索过程
ID_TYPE dstree::Allocator::simulate_dstree_search_for_query(
    ID_TYPE query_id, 
    ID_TYPE batch_i, 
    ID_TYPE error_i,
    const std::unordered_map<ID_TYPE, size_t>& node_id_to_index,
    std::shared_ptr<dstree::Node> root) {
  
  ID_TYPE hit_count = 0;
  VALUE_TYPE current_bsf = constant::MAX_VALUE;  // 初始化为最大值
  ID_TYPE total_leaf_nodes = 0;  // 统计访问的叶子节点总数
  
  // 添加调试信息（仅为第一个查询打印，避免输出过多）
  bool debug_print = (query_id == 0 && batch_i == 0 && error_i == 0);
  if (debug_print) {
    printf("\n=== 调试: simulate_dstree_search_for_query ===\n");
    printf("查询ID: %ld, 批次: %ld, 误差分位数: %ld\n", query_id, batch_i, error_i);
  }
  
  // 第一步：找到当前查询的target_node（k=1情况下就是最近的那个节点）
  const ID_TYPE K = config_.get().n_nearest_neighbor_;
  ID_TYPE target_node_id = -1;
  VALUE_TYPE min_distance = constant::MAX_VALUE;
  
  // 遍历所有叶子节点，找到距离最近的那个作为target_node
  for (size_t filter_idx = 0; filter_idx < filter_infos_.size(); ++filter_idx) {
    auto& filter_info = filter_infos_[filter_idx];
    auto& node = filter_info.node_;
    
    if (node.get().is_leaf()) {
      VALUE_TYPE true_distance = node.get().get_filter_nn_distance(query_id);
      if (true_distance < min_distance) {
        min_distance = true_distance;
        target_node_id = node.get().get_id();
      }
    }
  }
  
  if (debug_print) {
    printf("查询 %ld 的target_node_id: %ld, 真实距离: %.6f\n", query_id, target_node_id, min_distance);
  }
  
  // 第二步：模拟真实的dstree搜索过程，访问所有叶子节点来正确计算BSF
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
  local_leaf_min_heap.push(std::make_tuple(std::ref(*root), 0));
  
  // 按真实访问顺序遍历所有叶子节点
  while (!local_leaf_min_heap.empty()) {
    auto [node_to_visit, node2visit_lbdistance] = local_leaf_min_heap.top();
    local_leaf_min_heap.pop();
    
    if (node_to_visit.get().is_leaf()) {
      // 处理叶子节点
      ID_TYPE node_id = node_to_visit.get().get_id();
      auto map_it = node_id_to_index.find(node_id);
      
      if (map_it == node_id_to_index.end()) {
        continue;
      }
      
      size_t node_index = map_it->second;
      if (node_index >= filter_infos_.size()) {
        continue;
      }
      
      auto& filter_info = filter_infos_[node_index];
      auto& current_node = filter_info.node_;
      
      // 获取当前节点的真实距离
      VALUE_TYPE true_distance = current_node.get().get_filter_nn_distance(query_id);
      
      // 更新BSF（模拟真实搜索过程）
      if (true_distance < current_bsf) {
        current_bsf = true_distance;
      }
      
      total_leaf_nodes++;
      
      // 关键：只有当访问到target_node时，才进行剪枝判断
      if (node_id == target_node_id) {
        bool node_preserved = true;  // 默认保留
        
        if (current_node.get().has_active_filter()) {
          VALUE_TYPE abs_error = current_node.get().get_filter_batch_abs_error_interval_by_pos(batch_i, error_i);
          VALUE_TYPE pred_distance = current_node.get().get_filter_pred_distance(query_id);
          
          // 剪枝条件：如果预测距离减去误差大于当前BSF，则被剪枝
          if (pred_distance - abs_error > current_bsf) {
            node_preserved = false;  // 被剪枝
            if (debug_print) {
              printf("target_node %ld 被剪枝: pred_dist-error=%.6f > bsf=%.6f (访问顺序: %ld)\n", 
                     node_id, pred_distance - abs_error, current_bsf, total_leaf_nodes);
            }
          } else {
            if (debug_print) {
              printf("target_node %ld 保留: pred_dist-error=%.6f <= bsf=%.6f (访问顺序: %ld)\n", 
                     node_id, pred_distance - abs_error, current_bsf, total_leaf_nodes);
            }
          }
        } else {
          // 没有过滤器的节点总是被保留
          if (debug_print) {
            printf("target_node %ld 保留: 无过滤器 (访问顺序: %ld)\n", node_id, total_leaf_nodes);
          }
        }
        
        if (node_preserved) {
          hit_count = 1;  // k=1情况下，target_node被保留则hit_count=1
        } else {
          hit_count = 0;  // k=1情况下，target_node被剪枝则hit_count=0
        }
      }
      
    } else {
      // 处理内部节点，将子节点加入队列
      for (auto child_node : node_to_visit.get()) {
        VALUE_TYPE child_lower_bound = 0;  // 简化处理
        local_leaf_min_heap.push(std::make_tuple(child_node, child_lower_bound));
      }
    }
  }
  
  if (debug_print) {
    printf("总访问节点数: %ld\n", total_leaf_nodes);
    printf("target_node保留状态 (hit_count): %ld\n", hit_count);
    printf("当前查询的召回率: %.4f\n", (double)hit_count);
  }
  
  return hit_count;
}







// 保存(recall, coverage)对到CSV文件
RESPONSE dstree::Allocator::save_recall_coverage_pairs(
    const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs) {
    printf("\n开始保存(recall, coverage, error)三元组到CSV文件\n");
    // 检查输入数据是否为空
    if (error_recall_cov_pairs.empty()) {
        printf("错误: 没有可保存的三元组数据\n");
        return FAILURE;
    }
    ID_TYPE num_error_quantiles = error_recall_cov_pairs.size();
    std::string save_path = config_.get().save_path_;
    // 确保目录存在
    if (!save_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(save_path)) {
        printf("创建结果保存目录: %s\n", save_path.c_str());
        fs::create_directories(save_path);
      }
    }
    std::string lgbm_raw_data_dir = save_path + "/lgbm_raw_data";
    if (!fs::exists(lgbm_raw_data_dir)) {
      fs::create_directories(lgbm_raw_data_dir);
    }
    
    // 为每个过滤器分别保存一个CSV文件
    for (auto& filter_info : filter_infos_) {
        if (filter_info.node_.get().has_active_filter()) {
            ID_TYPE filter_id = filter_info.node_.get().get_id();
            auto& filter = filter_info.node_.get().get_filter().get();
            // 构造CSV文件名
            std::string csv_filename = lgbm_raw_data_dir + "/filter_" + std::to_string(filter_id) + "_raw_data.csv";
            // 打开CSV文件
            std::ofstream csv_file(csv_filename);
            if (!csv_file.is_open()) {
                printf("错误: 无法创建CSV文件 %s\n", csv_filename.c_str());
                continue;
            }
            
            // 直接使用error_recall_cov_pairs中的数据，不重新计算
            std::vector<std::vector<std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>> 
                batch_data(error_recall_cov_pairs[0].size(), std::vector<std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>(num_error_quantiles));
            
            // 直接从error_recall_cov_pairs读取recall和coverage
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
                    ERROR_TYPE recall = error_recall_cov_pairs[error_i][batch_i].first;
                    ERROR_TYPE coverage = error_recall_cov_pairs[error_i][batch_i].second;
                    ERROR_TYPE error = filter.get_batch_abs_error_interval_by_pos(batch_i, error_i);
                    
                    batch_data[batch_i][error_i] = std::make_tuple(recall, coverage, error);
                    // printf("batch_i=%ld, error_i=%ld, recall=%.3f, coverage=%.3f, error=%.6f\n", batch_i, error_i, recall, coverage, error);
                }
                
                // 按照recall升序排序当前误差分位的所有批次
                std::vector<std::pair<ID_TYPE, std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>> batch_with_index;
                for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
                    batch_with_index.push_back({batch_i, batch_data[batch_i][error_i]});
                }
                std::sort(batch_with_index.begin(), batch_with_index.end(),
                         [](const auto& a, const auto& b) {
                             return std::get<0>(a.second) < std::get<0>(b.second); // 按照recall排序
                         });
                // 更新排序后的数据
                for (ID_TYPE i = 0; i < batch_with_index.size(); ++i) {
                    batch_data[i][error_i] = batch_with_index[i].second;
                }
            }

            // 写入CSV标题行
            csv_file << "batch_id_sorted";
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                csv_file << ",recall,cov,actual error";
            }
            csv_file << std::endl;
            
            // 写入每个批次的数据
            for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[0].size(); ++batch_i) {
                csv_file << batch_i;
                
                // 遍历每个误差分位
                for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                    auto [recall, coverage, error] = batch_data[batch_i][error_i];
                    
                    // 写入召回率、覆盖率和误差值，保留4位小数
                    csv_file << "," << std::fixed << std::setprecision(4) << recall
                             << "," << std::fixed << std::setprecision(4) << coverage
                             << "," << std::fixed << std::setprecision(6) << error;
                }
                csv_file << std::endl;
            }
            
            // 关闭CSV文件
            csv_file.close();
            printf("已成功保存过滤器 %ld 的(recall, coverage, error)三元组到 %s\n", 
                   filter_id, csv_filename.c_str());
        }
    }
    
    return SUCCESS;
} 









// 计算每个误差分位数下的(recall, coverage)对并用于filter拟合
RESPONSE dstree::Allocator::calculate_recall_coverage_pairs() {
  // 优先使用模拟的完整搜索召回率数据，如果不存在则使用原始数据
  std::vector<std::vector<ERROR_TYPE>>* recalls_data = nullptr;
  
  if (!batch_validation_recalls_simulated_.empty()) {
    printf("使用模拟完整搜索的召回率数据进行计算\n");
    recalls_data = &batch_validation_recalls_simulated_;
  } else if (!batch_validation_recalls_.empty()) {
    printf("使用原始召回率数据进行计算\n");
    recalls_data = &batch_validation_recalls_;
  } else {
    printf("错误: 未找到批次召回率数据\n");
    return FAILURE;
  }
  
  ID_TYPE num_batches = recalls_data->size();
  ID_TYPE num_error_quantiles = (*recalls_data)[0].size();
  
  // 存储每个误差分位数下的(recall, coverage)对
  std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>> error_recall_cov_pairs(num_error_quantiles);
  
  // 遍历每个误差分位数
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    // 收集该误差分位数下所有批次的召回率， recalls是一列的内容
    std::vector<ERROR_TYPE> recalls;
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
      recalls.push_back((*recalls_data)[batch_i][error_i]);
    }
    
    // 对每个排序后的召回率计算覆盖率 (好像还没排序呢？)
    for (ID_TYPE j = 0; j < recalls.size(); ++j) {
      ERROR_TYPE min_recall = recalls[j];
      ID_TYPE satisfying_batches = 0;
      // 计算达到min_recall的批次数量
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        if ((*recalls_data)[batch_i][error_i] >= min_recall) {
            satisfying_batches++;
        }
      }     
      ERROR_TYPE coverage = static_cast<ERROR_TYPE>(satisfying_batches) / num_batches;
      error_recall_cov_pairs[error_i].emplace_back(min_recall, coverage);
    }
    // // 对该误差分位数下的(recall, coverage)对按recall升序排列
    // std::sort(error_recall_cov_pairs[error_i].begin(), error_recall_cov_pairs[error_i].end(),[](const std::pair<ERROR_TYPE, ERROR_TYPE>& a, const std::pair<ERROR_TYPE, ERROR_TYPE>& b) {
    //   return a.first < b.first;  // 按recall升序排列
    // });

  }
  
  // 打印和保存error_recall_cov_pairs矩阵
  printf("\n==== error_recall_cov_pairs矩阵 ====\n");
  printf("行表示pair索引，列表示error_i误差分位数，每个位置是(recall, coverage)对\n");
  printf("每列已按recall升序排序\n\n");
  
  // 计算最大行数（最大的pair数量）
  size_t max_pairs = 0;
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    max_pairs = std::max(max_pairs, error_recall_cov_pairs[error_i].size());
  }
  
  // 确保save_path存在
  std::string save_path = config_.get().save_path_;
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      fs::create_directories(save_path);
    }
  }
  

  // 保存(recall, coverage)对到CSV文件
  save_recall_coverage_pairs(error_recall_cov_pairs);
  // 在for循环之前初始化变量用于计算平均alpha
  std::vector<VALUE_TYPE> all_filter_alphas;
  size_t valid_filter_count = 0;
  
  std::string lgbm_data_dir = save_path + "/lgbm_data";
  namespace fs = boost::filesystem;
  if (!fs::exists(lgbm_data_dir)) {
    fs::create_directories(lgbm_data_dir);
    printf("创建LightGBM数据目录: %s\n", lgbm_data_dir.c_str());
  }
  
  // 为每个filter训练统一的二元回归模型
  for (auto& filter_info : filter_infos_) {
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      
      // printf("\n====================为节点 %ld 训练统一的二元回归模型===============\n", filter_info.node_.get().get_id());
      spdlog::info("\n================为节点 {} 训练统一的二元回归模型=================\n", filter_info.node_.get().get_id());
      // 收集所有误差分位数下的所有(recall, coverage)对和对应的误差位置
      std::vector<ERROR_TYPE> all_recalls;
      std::vector<ERROR_TYPE> all_coverages;
      std::vector<ID_TYPE> all_error_indices; // 仍然需要误差索引
      std::vector<ERROR_TYPE> all_errors;     // 新增：对应的实际误差值
      
      // 存储唯一的(recall-coverage)对 及{max_error_i, best_batch_id}
      std::unordered_map<std::string, std::pair<ID_TYPE, ID_TYPE>> unique_pairs_max_error; // 改为存储{max_error_i, best_batch_id}
      // 从所有误差分位数收集数据并去重
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
          const auto& [recall, coverage] = error_recall_cov_pairs[error_i][batch_i];
          // 创建唯一键
          std::string key = fmt::format("{:.6f}_{:.6f}", recall, coverage);
          // 检查是否已存在该(recall, coverage)对
          auto it = unique_pairs_max_error.find(key);
          if (it != unique_pairs_max_error.end()) {
            // 如果存在且当前误差索引大于已保存的，则更新
            if (error_i > it->second.first) {
              it->second = {error_i, batch_i};
            } else if (error_i == it->second.first) {
              // 如果误差索引相同，比较error值，取较大者
              VALUE_TYPE current_error = filter.get_batch_abs_error_interval_by_pos(batch_i, error_i);
              VALUE_TYPE existing_error = filter.get_batch_abs_error_interval_by_pos(it->second.second, error_i);
              if (current_error > existing_error) {
                it->second.second = batch_i;
              }
            }
          } else {
            // 如果不存在，则添加
            unique_pairs_max_error[key] = {error_i, batch_i};
          }
        }
      }
      

      // 使用唯一的(recall, coverage)对和它们的最大误差索引, 获取去重之后的(recall, coverage, error_value)对
      // printf("========= 利用unique_pairs_max_error开始生成all_recalls, all_coverages, all_error_indices, all_errors=========\n");
      for (const auto& [key, error_batch_pair] : unique_pairs_max_error) {
        // 从key中提取recall和coverage
        float recall, coverage;
        sscanf(key.c_str(), "%f_%f", &recall, &coverage);
        ID_TYPE max_error_i = error_batch_pair.first;
        ID_TYPE best_batch_i = error_batch_pair.second;
        
        all_recalls.push_back(recall);
        all_coverages.push_back(coverage);
        all_error_indices.push_back(max_error_i);
        
        // 获取对应的误差值（从过滤器中的特定batch）
        if (auto* cp = filter.get_conformal_predictor(); cp != nullptr) {
          VALUE_TYPE error_value = filter.get_batch_abs_error_interval_by_pos(best_batch_i, max_error_i);
          all_errors.push_back(error_value);
        }
      }

      // printf("========= 开始生成unique_pairs_max_error.size(): %zu =========\n", unique_pairs_max_error.size());
      std::vector<std::tuple<float, float, float>> unique_pairs_with_errors;
      for (const auto& [key, error_batch_pair] : unique_pairs_max_error) {
          float recall, coverage;
          sscanf(key.c_str(), "%f_%f", &recall, &coverage);
          ID_TYPE max_error_i = error_batch_pair.first;
          ID_TYPE best_batch_i = error_batch_pair.second;
          
          // 获取对应的误差值
          float error_value = 0.0f;
          if (auto* cp = filter.get_conformal_predictor(); cp != nullptr) {
              error_value = filter.get_batch_abs_error_interval_by_pos(best_batch_i, max_error_i);
          }
          unique_pairs_with_errors.emplace_back(recall, coverage, error_value);
          // printf("recall: %f, coverage: %f, error_value: %f (from batch_%ld, error_i_%ld)\n", recall, coverage, error_value, best_batch_i, max_error_i);
      }
      // 按照召回率排序
      std::sort(unique_pairs_with_errors.begin(), unique_pairs_with_errors.end());
      // 保存每个filter的三元组数据到单独的txt文件
      ID_TYPE filter_id = filter_info.node_.get().get_id();
      std::string filter_data_path = lgbm_data_dir + "/filter_" + std::to_string(filter_id) + "_triples.csv";
      std::ofstream filter_file(filter_data_path);
      
      if (filter_file.is_open()) {
        filter_file << "recall,coverage,error,is_test" << std::endl;
        // 使用80%的数据作为训练集，20%作为测试集
        size_t total_pairs = unique_pairs_with_errors.size();
        size_t test_size = static_cast<size_t>(total_pairs * 0.2);
        // 创建一个随机索引数组，用于随机选择测试样本
        std::vector<size_t> indices(total_pairs);
        for (size_t i = 0; i < total_pairs; ++i) {
          indices[i] = i;
        }
        // 随机打乱索引
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        // 创建一个集合，存储测试集索引
        std::unordered_set<size_t> test_indices;
        for (size_t i = 0; i < test_size; ++i) {
          test_indices.insert(indices[i]);
        }
        // 写入数据行
        for (size_t i = 0; i < unique_pairs_with_errors.size(); ++i) {
          const auto& [recall, coverage, error_value] = unique_pairs_with_errors[i];
          // 判断是训练集(0)还是测试集(1)
          int is_test = test_indices.count(i) > 0 ? 1 : 0;
          filter_file << std::fixed << std::setprecision(6) 
                     << recall << "," 
                     << coverage << "," 
                     << error_value << "," 
                     << is_test << std::endl;
        }
        
        filter_file.close();
        printf("已保存Filter %ld 的三元组数据到: %s (共 %zu 个三元组)\n", 
               filter_id, filter_data_path.c_str(), unique_pairs_with_errors.size());
      } else {
        printf("警告: 无法创建文件: %s\n", filter_data_path.c_str());
      }
      
      // 检查数据点数量是否足够
      const size_t min_required_samples = 1; // 最小样本数要求
      if (unique_pairs_with_errors.size() < min_required_samples) {
          printf("警告: 节点 %ld 的数据点数量(%zu)不足以训练模型(最少需要%zu个点)\n", 
                 filter_info.node_.get().get_id(), unique_pairs_with_errors.size(), min_required_samples);
          spdlog::warn("节点 {} 的数据点数量({})不足以训练模型(最少需要{}个点)", 
                      filter_info.node_.get().get_id(), unique_pairs_with_errors.size(), min_required_samples);
          continue; // 跳过训练
      }



      // 根据配置选择使用的模型类型
      if (config_.get().use_train_optimal_polynomial_model_) {
          // 使用二元回归模型
          printf("=======train_regression_model_for_recall_coverage=======\n");
          if (filter_info.node_.get().train_regression_model_for_recall_coverage(
                  all_recalls, all_coverages, all_error_indices, 
                  filter_info.node_.get().get_id()) != SUCCESS) {
              printf("警告: 节点 %ld 训练统一的二元回归模型失败\n", filter_info.node_.get().get_id());
              spdlog::error("警告: 节点 {} 训练统一的二元回归模型失败", filter_info.node_.get().get_id());
              continue;
          }

      } else {
          // 使用Eigen二次样条模型
          printf("----  fit_alglib_quadratic_spline ----\n");
          // 创建一个存储模型系数的向量  fit_eigen_quadratic_spline   fit_alglib_quadratic_spline
          std::vector<double> alglib_spline_coeffs;
          if (!all_errors.empty() && filter_info.node_.get().fit_eigen_quadratic_spline(
                  all_recalls, all_coverages, all_errors, 
                  alglib_spline_coeffs) != SUCCESS) {
              printf("警告: 节点 %ld alglib二次样条模型拟合失败\n", 
                    filter_info.node_.get().get_id());
              spdlog::error("警告: 节点 {} alglib二次样条模型拟合失败", 
                          filter_info.node_.get().get_id());
          }
          // 此时eigen_spline_coeffs已包含模型系数，但实际上不需要使用它
          // 因为函数内部已将系数保存到regression_coeffs_成员变量
      }


      // 使用训练好的模型进行预测和设置
      ERROR_TYPE target_recall = config_.get().filter_conformal_recall_;
      ERROR_TYPE target_coverage = config_.get().filter_conformal_coverage_;
      // 最后只使用用户指定的值实际设置alpha
      RESPONSE result_target = filter_info.node_.get().set_filter_abs_error_interval_by_recall_and_coverage(
          target_recall, target_coverage);
      if (result_target == SUCCESS) {
          VALUE_TYPE alpha = filter_info.node_.get().get_filter_abs_error_interval();
          all_filter_alphas.push_back(alpha);
          valid_filter_count++;
          spdlog::info("Filter {} 在 R={:.2f}, C={:.2f} 下的alpha值: {:.4f}", 
                  filter_info.node_.get().get_id(), target_recall, target_coverage, alpha);
          // printf("(Filter %ld 在 R=%.2f, C=%.2f, alpha=%.4f)\n", filter_info.node_.get().get_id(), target_recall, target_coverage, filter_info.node_.get().get_filter_abs_error_interval());  
      }
      
      // // 保存最后一个处理的filter的unique_pairs_with_errors
      // if (&filter_info == &filter_infos_.back() || std::distance(&filter_info, &filter_infos_.back()) == filter_infos_.size() - 1) {
      //   printf("\n\n==== 最后一个Filter (节点ID: %ld) 的排序后recall, coverage, error三元组 ====\n", 
      //          filter_info.node_.get().get_id());
      //   printf("No.  | Recall  | Coverage | Error\n");
      //   printf("---------------------------------\n");
      //   // 按recall从小到大排序并打印
      //   std::sort(unique_pairs_with_errors.begin(), unique_pairs_with_errors.end());
      //   for (size_t i = 0; i < unique_pairs_with_errors.size(); ++i) {
      //     const auto& [recall, coverage, error_value] = unique_pairs_with_errors[i];
      //     printf("%-4zu | %-7.4f | %-7.4f | %-7.6f\n", 
      //            i+1, recall, coverage, error_value);
      //   }
      //   printf("\n总共 %zu 个唯一三元组\n\n", unique_pairs_with_errors.size());
      // }

    }
  }

  // 在循环结束后计算平均alpha
  if (!all_filter_alphas.empty()) {
    VALUE_TYPE total_alpha = 0.0;
    for (auto alpha : all_filter_alphas){
      total_alpha += alpha;
    }
    VALUE_TYPE average_alpha = total_alpha / all_filter_alphas.size();
    printf("\n 满足target recall和coverage的平均预测alpha值: %.4f\n", average_alpha);
  }
  // printf("回归模型训练完成，验证数据结构完整性\n");
  // 在原有代码结束后添加
  // 保存所有过滤器的预测alpha值
  // std::string save_path = config_.get().save_path_;
  // if (!save_path.empty()) {
  //   namespace fs = boost::filesystem;
  //   if (!fs::exists(save_path)) {
  //     fs::create_directories(save_path);
  //   }
  // }
  
  // std::string alpha_values_path = save_path + "/predicted_alphas.csv";
  // if (save_predicted_alphas(alpha_values_path) == SUCCESS) {
  //   printf("已保存所有过滤器的预测alpha值到: %s\n", alpha_values_path.c_str());
  // } else {
  //   printf("保存预测alpha值失败\n");
  // }

  
  return SUCCESS;
}











// 新增：保存所有filter的预测alpha值到文件
RESPONSE dstree::Allocator::save_predicted_alphas(const std::string& filepath) {
  namespace fs = boost::filesystem;
  fs::path dir_path = fs::path(filepath).parent_path();
  if (!fs::exists(dir_path)) {
    fs::create_directories(dir_path);
  }
  std::ofstream alpha_file(filepath);
  if (!alpha_file.is_open()) {
    printf("错误: 无法创建alpha值文件 %s\n", filepath.c_str());
    return FAILURE;
  }
  // 获取目标召回率和覆盖率
  ERROR_TYPE target_recall = config_.get().filter_conformal_recall_;
  ERROR_TYPE target_coverage = config_.get().filter_conformal_coverage_;
  // 写入表头
  alpha_file << "node_id,method,target_recall,target_coverage,predicted_alpha,model_coefficients" << std::endl;
  // 遍历所有filter
  for (auto& filter_info : filter_infos_) {
    if (filter_info.node_.get().has_active_filter()) {
      ID_TYPE node_id = filter_info.node_.get().get_id();
      auto& filter = filter_info.node_.get().get_filter().get();
      // 获取当前alpha值
      VALUE_TYPE current_alpha = filter_info.node_.get().get_filter_abs_error_interval();
      // 记录当前使用的训练方法
      std::string method = config_.get().use_train_optimal_polynomial_model_ ? "polynomial" : "spline";
      // 获取模型系数
      std::vector<double> coeffs;
      
      if (auto* cp = filter.get_conformal_predictor()) {
        coeffs = cp->get_regression_coefficients();
      }
      // 将系数转换为字符串
      std::string coeffs_str = "[";
      for (size_t i = 0; i < coeffs.size(); ++i) {
        coeffs_str += fmt::format("{:.6f}", coeffs[i]);
        if (i < coeffs.size() - 1) {
          coeffs_str += ",";
        }
      }
      coeffs_str += "]";
      // 写入数据行
      alpha_file << node_id << ","
                << method << ","
                << std::fixed << std::setprecision(4) << target_recall << ","
                << std::fixed << std::setprecision(4) << target_coverage << ","
                << std::fixed << std::setprecision(6) << current_alpha << ","
                << coeffs_str << std::endl;
    }
  }
  alpha_file.close();
  // printf("成功保存所有filter的预测alpha值到: %s\n", filepath.c_str());
  return SUCCESS;
}






// 模拟完整dstree搜索过程，重新计算准确的recall
RESPONSE dstree::Allocator::simulate_full_search_for_recall_alpha_based(std::shared_ptr<dstree::Node> root) {
  printf("\n ------Allocator::simulate_full_search_for_recall_alpha_based ------ \n");
  
  // 1. 从Node ID快速找到对应的Filter index， Filter Index = 该Filter在filter_infos_数组中的位置， Filter ID = Node ID（它们是同一个值）
  std::unordered_map<ID_TYPE, size_t> node_id_to_index;
  for (size_t i = 0; i < filter_infos_.size(); ++i) {
    node_id_to_index[filter_infos_[i].node_.get().get_id()] = i;
  }
  printf("总共有 %zu 个filter\n", filter_infos_.size());
  spdlog::info("总共有 {} 个filter", filter_infos_.size());
  const ID_TYPE K = config_.get().n_nearest_neighbor_; 
  
  // 2. 获取校准集批次信息
  ID_TYPE num_batches = 0;
  ID_TYPE examples_per_batch = 0;
  std::vector<std::vector<ID_TYPE>> batch_query_ids;
  
  bool found_calibration_info = false;
  for (size_t i = 0; i < filter_infos_.size() && !found_calibration_info; ++i) {
    auto& filter_info = filter_infos_[i];
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      if (!filter.get_batch_calib_query_ids().empty()) {
        batch_query_ids = filter.get_batch_calib_query_ids();
        num_batches = batch_query_ids.size();
        examples_per_batch = batch_query_ids[0].size();
        found_calibration_info = true;
        printf("获取校准批次信息: %ld 批次, 每批 %ld 样本\n", num_batches, examples_per_batch);
      }
    }
  }
  
  if (!found_calibration_info) {
    printf("错误: 未找到校准批次信息\n");
    return FAILURE;
  }
  
  ID_TYPE num_error_quantiles = filter_infos_[0].node_.get().get_alphas_size();
  printf("误差分位数数量: %ld\n", num_error_quantiles);
  
  // 添加调试信息：打印关键配置参数
  printf("关键配置参数:\n");
  printf("  K (最近邻数量): %ld\n", K);
  printf("  批次数量: %ld\n", num_batches);
  printf("  每批样本数: %ld\n", examples_per_batch);
  printf("  叶子节点总数: %zu\n", filter_infos_.size());
  printf("  每批总KNN数 (K * batch_size): %ld\n", K * examples_per_batch);

  if (!is_recall_calculated_) { // 初始为false，执行内部代码

    // 3. 重新初始化召回率存储结构
    batch_validation_recalls_simulated_.clear();
    batch_validation_recalls_simulated_.resize(num_batches);
    
    // 4. 外层循环: 遍历误差分位数
    for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
      // printf("处理误差分位数 %ld/%ld\n", error_i + 1, num_error_quantiles);
      
      // 5. 中层循环: 遍历校准集批次
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        const std::vector<ID_TYPE>& current_batch_query_ids = batch_query_ids[batch_i];
        ID_TYPE batch_size = current_batch_query_ids.size();
        ID_TYPE total_hit_count = 0;
        const ID_TYPE batch_total_knn = K * batch_size;
        
        // 6. 内层循环: 遍历当前批次的每个查询
        for (ID_TYPE query_idx = 0; query_idx < batch_size; ++query_idx) {
          ID_TYPE current_query_id = current_batch_query_ids[query_idx] - 1;
          
          // 7. 模拟完整的dstree搜索过程
          ID_TYPE hit_count = simulate_dstree_search_alpha_based_for_query(
              current_query_id, batch_i, error_i, node_id_to_index, root);
          
          total_hit_count += hit_count;
        }
        
        // 计算当前批次、当前误差分位数下的召回率
        ERROR_TYPE recall = static_cast<ERROR_TYPE>(total_hit_count) / batch_total_knn;
        batch_validation_recalls_simulated_[batch_i].push_back(recall);
        
        // 添加调试信息：打印详细的计算过程
        // printf("  批次 %ld/%ld, 误差分位数 %ld/%ld: total_hit_count=%ld, batch_total_knn=%ld, 召回率=%.4f\n", 
        //        batch_i + 1, num_batches, error_i + 1, num_error_quantiles, 
        //        total_hit_count, batch_total_knn, recall);
        
        // 检查召回率是否超出合理范围
        if (recall > 1.0) {
          printf("警告: 召回率 %.4f 超过1.0，可能存在计算错误!\n", recall);
          printf("       total_hit_count=%ld, batch_total_knn=%ld\n", total_hit_count, batch_total_knn);
        }
        
        // 打印进度信息
        // if (batch_i % 10 == 0) {
        //   // printf("  批次 %ld/%ld, 召回率: %.4f\n", batch_i + 1, num_batches, recall);
        // }
      }
    }

    if (calculate_alpha_based_recall_coverage_pairs() != SUCCESS) {
        printf("警告: 计算(recall, coverage)对失败\n");
    }
    is_recall_calculated_ = true;
    
  }
    // 安全释放临时变量，避免析构时的问题
  node_id_to_index.clear();
  batch_query_ids.clear();

  printf("完整搜索模拟完成\n");
  return SUCCESS;
}



// 模拟单个查询的dstree搜索过程
ID_TYPE dstree::Allocator::simulate_dstree_search_alpha_based_for_query(
    ID_TYPE query_id, 
    ID_TYPE batch_i, 
    ID_TYPE error_i,
    const std::unordered_map<ID_TYPE, size_t>& node_id_to_index,
    std::shared_ptr<dstree::Node> root) {
  
  ID_TYPE hit_count = 0;
  VALUE_TYPE current_bsf = constant::MAX_VALUE;  // 初始化为最大值
  ID_TYPE total_leaf_nodes = 0;  // 统计访问的叶子节点总数
  
  // 添加调试信息（仅为第一个查询打印，避免输出过多）
  bool debug_print = (query_id == 0 && batch_i == 0 && error_i == 0);
  if (debug_print) {
    printf("\n=== 调试: simulate_dstree_search_for_query ===\n");
    printf("查询ID: %ld, 批次: %ld, 误差分位数: %ld\n", query_id, batch_i, error_i);
  }
  
  // 第一步：找到当前查询的target_node（k=1情况下就是最近的那个节点）
  const ID_TYPE K = config_.get().n_nearest_neighbor_;
  ID_TYPE target_node_id = -1;
  VALUE_TYPE min_distance = constant::MAX_VALUE;
  
  // 遍历所有叶子节点，找到距离最近的那个作为target_node
  for (size_t filter_idx = 0; filter_idx < filter_infos_.size(); ++filter_idx) {
    auto& filter_info = filter_infos_[filter_idx];
    auto& node = filter_info.node_;
    
    if (node.get().is_leaf()) {
      VALUE_TYPE true_distance = node.get().get_filter_nn_distance(query_id);
      if (true_distance < min_distance) {
        min_distance = true_distance;
        target_node_id = node.get().get_id();
      }
    }
  }
  
  if (debug_print) {
    printf("查询 %ld 的target_node_id: %ld, 真实距离: %.6f\n", query_id, target_node_id, min_distance);
  }
  
  // 第二步：模拟真实的dstree搜索过程，访问所有叶子节点来正确计算BSF
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
  local_leaf_min_heap.push(std::make_tuple(std::ref(*root), 0));
  
  // 按真实访问顺序遍历所有叶子节点
  while (!local_leaf_min_heap.empty()) {
    auto [node_to_visit, node2visit_lbdistance] = local_leaf_min_heap.top();
    local_leaf_min_heap.pop();
    
    if (node_to_visit.get().is_leaf()) {
      // 处理叶子节点
      ID_TYPE node_id = node_to_visit.get().get_id();
      auto map_it = node_id_to_index.find(node_id);
      
      if (map_it == node_id_to_index.end()) {
        continue;
      }
      
      size_t node_index = map_it->second;
      if (node_index >= filter_infos_.size()) {
        continue;
      }
      
      auto& filter_info = filter_infos_[node_index];
      auto& current_node = filter_info.node_;
      
      // 获取当前节点的真实距离
      VALUE_TYPE true_distance = current_node.get().get_filter_nn_distance(query_id);
      
      // 更新BSF（模拟真实搜索过程）
      if (true_distance < current_bsf) {
        current_bsf = true_distance;
      }
      
      total_leaf_nodes++;
      
      // 关键：只有当访问到target_node时，才进行剪枝判断
      if (node_id == target_node_id) {
        bool node_preserved = true;  // 默认保留
        
        if (current_node.get().has_active_filter()) {
          VALUE_TYPE abs_error = current_node.get().get_filter_abs_error_interval_by_pos(error_i);
          VALUE_TYPE pred_distance = current_node.get().get_filter_pred_distance(query_id);
          
          // 剪枝条件：如果预测距离减去误差大于当前BSF，则被剪枝
          if (pred_distance - abs_error > current_bsf) {
            node_preserved = false;  // 被剪枝
            if (debug_print) {
              printf("target_node %ld 被剪枝: pred_dist-error=%.6f > bsf=%.6f (访问顺序: %ld)\n", 
                     node_id, pred_distance - abs_error, current_bsf, total_leaf_nodes);
            }
          } else {
            if (debug_print) {
              printf("target_node %ld 保留: pred_dist-error=%.6f <= bsf=%.6f (访问顺序: %ld)\n", 
                     node_id, pred_distance - abs_error, current_bsf, total_leaf_nodes);
            }
          }
        } else {
          // 没有过滤器的节点总是被保留
          if (debug_print) {
            printf("target_node %ld 保留: 无过滤器 (访问顺序: %ld)\n", node_id, total_leaf_nodes);
          }
        }
        
        if (node_preserved) {
          hit_count = 1;  // k=1情况下，target_node被保留则hit_count=1
        } else {
          hit_count = 0;  // k=1情况下，target_node被剪枝则hit_count=0
        }
      }
      
    } else {
      // 处理内部节点，将子节点加入队列
      for (auto child_node : node_to_visit.get()) {
        VALUE_TYPE child_lower_bound = 0;  // 简化处理
        local_leaf_min_heap.push(std::make_tuple(child_node, child_lower_bound));
      }
    }
  }
  
  if (debug_print) {
    printf("总访问节点数: %ld\n", total_leaf_nodes);
    printf("target_node保留状态 (hit_count): %ld\n", hit_count);
    printf("当前查询的召回率: %.4f\n", (double)hit_count);
  }
  
  return hit_count;
}





// 计算每个误差分位数下的(recall, coverage)对并用于filter拟合
RESPONSE dstree::Allocator::calculate_alpha_based_recall_coverage_pairs() {
  // 优先使用模拟的完整搜索召回率数据，如果不存在则使用原始数据
  std::vector<std::vector<ERROR_TYPE>>* recalls_data = nullptr;
  
  if (!batch_validation_recalls_simulated_.empty()) {
    printf("使用模拟完整搜索的召回率数据进行计算\n");
    recalls_data = &batch_validation_recalls_simulated_;
  } else if (!batch_validation_recalls_.empty()) {
    printf("使用原始召回率数据进行计算\n");
    recalls_data = &batch_validation_recalls_;
  } else {
    printf("错误: 未找到批次召回率数据\n");
    return FAILURE;
  }
  
  ID_TYPE num_batches = recalls_data->size();
  ID_TYPE num_error_quantiles = (*recalls_data)[0].size();
  
  // 存储每个误差分位数下的(recall, coverage)对
  std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>> error_recall_cov_pairs(num_error_quantiles);
  
  // 遍历每个误差分位数
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    // 收集该误差分位数下所有批次的召回率， recalls是一列的内容
    std::vector<ERROR_TYPE> recalls;
    for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
      recalls.push_back((*recalls_data)[batch_i][error_i]);
    }
    
    // 对每个排序后的召回率计算覆盖率 (好像还没排序呢？)
    for (ID_TYPE j = 0; j < recalls.size(); ++j) {
      ERROR_TYPE min_recall = recalls[j];
      ID_TYPE satisfying_batches = 0;
      // 计算达到min_recall的批次数量
      for (ID_TYPE batch_i = 0; batch_i < num_batches; ++batch_i) {
        if ((*recalls_data)[batch_i][error_i] >= min_recall) {
            satisfying_batches++;
        }
      }     
      ERROR_TYPE coverage = static_cast<ERROR_TYPE>(satisfying_batches) / num_batches;
      error_recall_cov_pairs[error_i].emplace_back(min_recall, coverage);
    }

  }
  
  // 打印和保存error_recall_cov_pairs矩阵
  printf("\n==== error_recall_cov_pairs矩阵 ====\n");
  printf("行表示pair索引，列表示error_i误差分位数，每个位置是(recall, coverage)对\n");
  printf("每列已按recall升序排序\n\n");
  
  // 计算最大行数（最大的pair数量）
  size_t max_pairs = 0;
  for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
    max_pairs = std::max(max_pairs, error_recall_cov_pairs[error_i].size());
  }
  
  // 确保save_path存在
  std::string save_path = config_.get().save_path_;
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      fs::create_directories(save_path);
    }
  }
  

  // 保存(recall, coverage)对到CSV文件
  save_recall_alpha_based_coverage_pairs(error_recall_cov_pairs);





  // 在for循环之前初始化变量用于计算平均alpha
  std::vector<VALUE_TYPE> all_filter_alphas;
  size_t valid_filter_count = 0;
  
  std::string lgbm_data_dir = save_path + "/lgbm_data";
  namespace fs = boost::filesystem;
  if (!fs::exists(lgbm_data_dir)) {
    fs::create_directories(lgbm_data_dir);
    printf("创建LightGBM数据目录: %s\n", lgbm_data_dir.c_str());
  }
  
  // 为每个filter训练统一的二元回归模型
  for (auto& filter_info : filter_infos_) {
    if (filter_info.node_.get().has_active_filter()) {
      auto& filter = filter_info.node_.get().get_filter().get();
      
      // printf("\n====================为节点 %ld 训练统一的二元回归模型===============\n", filter_info.node_.get().get_id());
      spdlog::info("\n================为节点 {} 训练统一的二元回归模型=================\n", filter_info.node_.get().get_id());
      // 收集所有误差分位数下的所有(recall, coverage)对和对应的误差位置
      std::vector<ERROR_TYPE> all_recalls;
      std::vector<ERROR_TYPE> all_coverages;
      std::vector<ID_TYPE> all_error_indices; // 仍然需要误差索引
      std::vector<ERROR_TYPE> all_errors;     // 新增：对应的实际误差值
      
      // 存储唯一的(recall-coverage)对及其最大error_i
      std::unordered_map<std::string, ID_TYPE> unique_pairs_max_error; // 改为只存储max_error_i
      // 从所有误差分位数收集数据并去重
      for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
        for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
          const auto& [recall, coverage] = error_recall_cov_pairs[error_i][batch_i];
          // 创建唯一键
          std::string key = fmt::format("{:.6f}_{:.6f}", recall, coverage);
          // 检查是否已存在该(recall, coverage)对
          auto it = unique_pairs_max_error.find(key);
          if (it != unique_pairs_max_error.end()) {
            // 如果存在且当前误差索引大于已保存的，则更新
            if (error_i > it->second) {
              it->second = error_i;
            }
          } else {
            // 如果不存在，则添加
            unique_pairs_max_error[key] = error_i;
          }
        }
      }
      
      // 使用唯一的(recall, coverage)对和它们的最大误差索引, 获取去重之后的(recall, coverage, error_value)对
      // printf("========= 利用unique_pairs_max_error开始生成all_recalls, all_coverages, all_error_indices, all_errors=========\n");
      for (const auto& [key, max_error_i] : unique_pairs_max_error) {
        // 从key中提取recall和coverage
        float recall, coverage;
        sscanf(key.c_str(), "%f_%f", &recall, &coverage);
        
        all_recalls.push_back(recall);
        all_coverages.push_back(coverage);
        all_error_indices.push_back(max_error_i);
        
        VALUE_TYPE error_value = filter_info.node_.get().get_filter_abs_error_interval_by_pos(max_error_i);
        all_errors.push_back(error_value);

        // 获取对应的误差值（从alphas_数组中）
        // if (auto* cp = filter.get_conformal_predictor(); cp != nullptr) {
        //   VALUE_TYPE error_value = cp->get_alpha_by_pos(max_error_i);
        //   all_errors.push_back(error_value);
        // }
      }

      // printf("========= 开始生成unique_pairs_max_error.size(): %zu =========\n", unique_pairs_max_error.size());
      std::vector<std::tuple<float, float, float>> unique_pairs_with_errors;
      for (const auto& [key, max_error_i] : unique_pairs_max_error) {
          float recall, coverage;
          sscanf(key.c_str(), "%f_%f", &recall, &coverage);
          VALUE_TYPE error_value = filter_info.node_.get().get_filter_abs_error_interval_by_pos(max_error_i);
          unique_pairs_with_errors.emplace_back(recall, coverage, error_value);
          // printf("recall: %f, coverage: %f, error_value: %f (from alphas_[%ld])\n", recall, coverage, error_value, max_error_i);
      }
      // 按照召回率排序
      std::sort(unique_pairs_with_errors.begin(), unique_pairs_with_errors.end());
      // 保存每个filter的三元组数据到单独的txt文件
      ID_TYPE filter_id = filter_info.node_.get().get_id();
      std::string filter_data_path = lgbm_data_dir + "/filter_" + std::to_string(filter_id) + "_triples.csv";
      std::ofstream filter_file(filter_data_path);
      
      if (filter_file.is_open()) {
        filter_file << "recall,coverage,error,is_test" << std::endl;
        // 使用80%的数据作为训练集，20%作为测试集
        size_t total_pairs = unique_pairs_with_errors.size();
        size_t test_size = static_cast<size_t>(total_pairs * 0.3);
        // 创建一个随机索引数组，用于随机选择测试样本
        std::vector<size_t> indices(total_pairs);
        for (size_t i = 0; i < total_pairs; ++i) {
          indices[i] = i;
        }
        // 随机打乱索引
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
        // 创建一个集合，存储测试集索引
        std::unordered_set<size_t> test_indices;
        for (size_t i = 0; i < test_size; ++i) {
          test_indices.insert(indices[i]);
        }
        // 写入数据行
        for (size_t i = 0; i < unique_pairs_with_errors.size(); ++i) {
          const auto& [recall, coverage, error_value] = unique_pairs_with_errors[i];
          // 判断是训练集(0)还是测试集(1)
          int is_test = test_indices.count(i) > 0 ? 1 : 0;
          filter_file << std::fixed << std::setprecision(6) 
                     << recall << "," 
                     << coverage << "," 
                     << error_value << "," 
                     << is_test << std::endl;
        }
        
        filter_file.close();
        // printf("已保存Filter %ld 的三元组数据到: %s (共 %zu 个三元组)\n", 
        //        filter_id, filter_data_path.c_str(), unique_pairs_with_errors.size());
      } else {
        printf("警告: 无法创建文件: %s\n", filter_data_path.c_str());
      }
      
      // 检查数据点数量是否足够
      const size_t min_required_samples = 1; // 最小样本数要求
      if (unique_pairs_with_errors.size() < min_required_samples) {
          printf("警告: 节点 %ld 的数据点数量(%zu)不足以训练模型(最少需要%zu个点)\n", 
                 filter_info.node_.get().get_id(), unique_pairs_with_errors.size(), min_required_samples);
          spdlog::warn("节点 {} 的数据点数量({})不足以训练模型(最少需要{}个点)", 
                      filter_info.node_.get().get_id(), unique_pairs_with_errors.size(), min_required_samples);
          continue; // 跳过训练
      }



      // 根据配置选择使用的模型类型
      if (config_.get().use_train_optimal_polynomial_model_) {
          // 使用二元回归模型
          printf("=======train_regression_model_for_recall_coverage=======\n");
          if (filter_info.node_.get().train_regression_model_for_recall_coverage(
                  all_recalls, all_coverages, all_error_indices, 
                  filter_info.node_.get().get_id()) != SUCCESS) {
              printf("警告: 节点 %ld 训练统一的二元回归模型失败\n", filter_info.node_.get().get_id());
              spdlog::error("警告: 节点 {} 训练统一的二元回归模型失败", filter_info.node_.get().get_id());
              continue;
          }

      } else {
          // 使用Eigen二次样条模型
          // printf("----  fit_alglib_quadratic_spline ----\n");
          // 创建一个存储模型系数的向量  fit_eigen_quadratic_spline   fit_alglib_quadratic_spline
          std::vector<double> alglib_spline_coeffs;
          if (!all_errors.empty() && filter_info.node_.get().fit_eigen_quadratic_spline(
                  all_recalls, all_coverages, all_errors, 
                  alglib_spline_coeffs) != SUCCESS) {
              printf("警告: 节点 %ld alglib二次样条模型拟合失败\n", 
                    filter_info.node_.get().get_id());
              spdlog::error("警告: 节点 {} alglib二次样条模型拟合失败", 
                          filter_info.node_.get().get_id());
          }
          // 此时eigen_spline_coeffs已包含模型系数，但实际上不需要使用它
          // 因为函数内部已将系数保存到regression_coeffs_成员变量
      }


      // 使用训练好的模型进行预测和设置
      ERROR_TYPE target_recall = config_.get().filter_conformal_recall_;
      ERROR_TYPE target_coverage = config_.get().filter_conformal_coverage_;
      // 最后只使用用户指定的值实际设置alpha
      RESPONSE result_target = filter_info.node_.get().set_filter_abs_error_interval_by_recall_and_coverage(
          target_recall, target_coverage);
      if (result_target == SUCCESS) {
          VALUE_TYPE alpha = filter_info.node_.get().get_filter_abs_error_interval();
          all_filter_alphas.push_back(alpha);
          valid_filter_count++;
          spdlog::info("Filter {} 在 R={:.2f}, C={:.2f} 下的alpha值: {:.4f}", 
                  filter_info.node_.get().get_id(), target_recall, target_coverage, alpha);
          // printf("(Filter %ld 在 R=%.2f, C=%.2f, alpha=%.4f)\n", filter_info.node_.get().get_id(), target_recall, target_coverage, filter_info.node_.get().get_filter_abs_error_interval());  
      }
      
    }
  }

  // 在循环结束后计算平均alpha
  if (!all_filter_alphas.empty()) {
    VALUE_TYPE total_alpha = 0.0;
    for (auto alpha : all_filter_alphas){
      total_alpha += alpha;
    }
    VALUE_TYPE average_alpha = total_alpha / all_filter_alphas.size();
    // printf("\n 满足target recall和coverage的平均预测alpha值: %.4f\n", average_alpha);
  }


  
  return SUCCESS;
}






// 保存(recall, coverage)对到CSV文件
RESPONSE dstree::Allocator::save_recall_alpha_based_coverage_pairs(
    const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs) {
    // printf("\n开始保存(recall, coverage, error)三元组到CSV文件\n");
    // 检查输入数据是否为空
    if (error_recall_cov_pairs.empty()) {
        printf("错误: 没有可保存的三元组数据\n");
        return FAILURE;
    }
    ID_TYPE num_error_quantiles = error_recall_cov_pairs.size();
    std::string save_path = config_.get().save_path_;
    // 确保目录存在
    if (!save_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(save_path)) {
        printf("创建结果保存目录: %s\n", save_path.c_str());
        fs::create_directories(save_path);
      }
    }
    // 创建子文件夹 lgbm_raw_data
    std::string lgbm_raw_data_dir = save_path + "/lgbm_raw_data";
    if (!fs::exists(lgbm_raw_data_dir)) {
      fs::create_directories(lgbm_raw_data_dir);
    }
    
    // 为每个过滤器分别保存一个CSV文件
    for (auto& filter_info : filter_infos_) {
        if (filter_info.node_.get().has_active_filter()) {
            ID_TYPE filter_id = filter_info.node_.get().get_id();
            auto& filter = filter_info.node_.get().get_filter().get();
            // 构造CSV文件名
            std::string csv_filename = lgbm_raw_data_dir + "/filter_" + std::to_string(filter_id) + "_raw_data.csv";
            // 打开CSV文件
            std::ofstream csv_file(csv_filename);
            if (!csv_file.is_open()) {
                printf("错误: 无法创建CSV文件 %s\n", csv_filename.c_str());
                continue;
            }
            
            // 直接使用error_recall_cov_pairs中的数据，不重新计算
            std::vector<std::vector<std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>> 
                batch_data(error_recall_cov_pairs[0].size(), std::vector<std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>(num_error_quantiles));
            
            // 直接从error_recall_cov_pairs读取recall和coverage
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
                    ERROR_TYPE recall = error_recall_cov_pairs[error_i][batch_i].first;
                    ERROR_TYPE coverage = error_recall_cov_pairs[error_i][batch_i].second;
                    // error现在换成alphas中的误差
                    ERROR_TYPE error = filter_info.node_.get().get_filter_abs_error_interval_by_pos(error_i);
                    
                    batch_data[batch_i][error_i] = std::make_tuple(recall, coverage, error);
                    // printf("batch_i=%ld, error_i=%ld, recall=%.3f, coverage=%.3f, error=%.6f\n", batch_i, error_i, recall, coverage, error);
                }
                
                // 按照recall升序排序当前误差分位的所有批次
                std::vector<std::pair<ID_TYPE, std::tuple<ERROR_TYPE, ERROR_TYPE, ERROR_TYPE>>> batch_with_index;
                for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[error_i].size(); ++batch_i) {
                    batch_with_index.push_back({batch_i, batch_data[batch_i][error_i]});
                }
                std::sort(batch_with_index.begin(), batch_with_index.end(),
                         [](const auto& a, const auto& b) {
                             return std::get<0>(a.second) < std::get<0>(b.second); // 按照recall排序
                         });
                // 更新排序后的数据
                for (ID_TYPE i = 0; i < batch_with_index.size(); ++i) {
                    batch_data[i][error_i] = batch_with_index[i].second;
                }
            }

            // 写入CSV标题行
            csv_file << "batch_id_sorted";
            for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                csv_file << ",recall,cov,actual error";
            }
            csv_file << std::endl;
            
            // 写入每个批次的数据
            for (ID_TYPE batch_i = 0; batch_i < error_recall_cov_pairs[0].size(); ++batch_i) {
                csv_file << batch_i;
                
                // 遍历每个误差分位
                for (ID_TYPE error_i = 0; error_i < num_error_quantiles; ++error_i) {
                    auto [recall, coverage, error] = batch_data[batch_i][error_i];
                    
                    // 写入召回率、覆盖率和误差值，保留4位小数
                    csv_file << "," << std::fixed << std::setprecision(4) << recall
                             << "," << std::fixed << std::setprecision(4) << coverage
                             << "," << std::fixed << std::setprecision(6) << error;
                }
                csv_file << std::endl;
            }
            
            // 关闭CSV文件
            csv_file.close();
            // printf("已成功保存过滤器 %ld 的(recall, coverage, error)三元组到 %s\n", 
            //        filter_id, csv_filename.c_str());
        }
    }
    
    return SUCCESS;
} 

