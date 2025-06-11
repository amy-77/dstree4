//
// Created by Qitong Wang on 2023/2/22.
// Copyright (c) 2023 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_SRC_EXEC_FILTER_ALLOCATOR_H_
#define DSTREE_SRC_EXEC_FILTER_ALLOCATOR_H_

#include <vector>
#include <map>
#include <tuple>
#include <unordered_map>
#include <queue>
#include <memory>

#include "global.h"
#include "config.h"
#include "models.h"
#include "node.h"

namespace upcite {
namespace dstree {

// 类型定义，与index.h保持一致
using NODE_DISTNCE = std::tuple<std::reference_wrapper<dstree::Node>, VALUE_TYPE>;

struct FilterInfo {
 public:
  explicit FilterInfo(Node &node) :
      node_(node),
      model_setting(upcite::MODEL_SETTING_PLACEHOLDER_REF) {
    score = -1;

    external_pruning_probability_ = -1;
  };
  ~FilterInfo() = default;

  VALUE_TYPE score;
  std::reference_wrapper<MODEL_SETTING> model_setting;
  std::reference_wrapper<Node> node_;
  VALUE_TYPE external_pruning_probability_; // d_bsf < d_lb
//  VALUE_TYPE pruning_probability_; // d_bsf < d_p
//  VALUE_TYPE false_pruning_probability_; // d_nn < d_bsf < d_p
};


static bool compDecreFilterScore(dstree::FilterInfo &filter_info_1, dstree::FilterInfo &filter_info_2) {
  return filter_info_1.score > filter_info_2.score;
}

static bool compDecreFilterNSeries(dstree::FilterInfo &filter_info_1, dstree::FilterInfo &filter_info_2) {
  return filter_info_1.node_.get().get_size() > filter_info_2.node_.get().get_size();
}

class Allocator {
 public:
  explicit Allocator(Config &config,
                     ID_TYPE nfilters = -1);

  ~Allocator() = default;
  // 注入训练数据
  // void set_training_data(const std::vector<dstree::Answers>& train_answers);

  RESPONSE push_filter_info(const FilterInfo &filter_info);

  RESPONSE assign();
  RESPONSE reassign();
  RESPONSE calculate_recall_coverage_pairs();

  // RESPONSE set_batch_confidence_from_recall(const std::vector<upcite::dstree::Answers>& train_answers);
  // RESPONSE set_batch_confidence_from_recall(const std::vector<Answers>& train_answers);
  RESPONSE set_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);
  // 保存(recall, coverage)对到CSV文件
  RESPONSE save_recall_coverage_pairs(
      const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs);
  // 新增：保存(recall, coverage, error)三元组到CSV文件，针对每个filter的每个batch
  RESPONSE save_recall_coverage_error_pairs(
      const std::vector<std::vector<std::pair<ERROR_TYPE, ERROR_TYPE>>>& error_recall_cov_pairs);
  // 新增：多批次校准集的置信区间设置函数
  RESPONSE set_batch_confidence_from_recall(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);
  
  // 新增：保存带批次召回率的置信区间计算函数
  RESPONSE set_batch_confidence_from_recall_save(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);
  
  /**
   * 记录校准集所有批次的query的真实knn节点对应的真实距离、minbsf、pred_distance、abs_error和true_error
   * 
   * @param query_knn_nodes 查询ID到节点分布的映射，格式为<查询ID, <节点ID, 该节点下的真实KNN数量>>
   * @return RESPONSE 操作成功返回SUCCESS，否则返回FAILURE
   */
  RESPONSE document_cp_dist(const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);




  // 新增：保存节点访问和减枝统计数据
  RESPONSE save_node_access_stats(
      const std::vector<std::vector<ID_TYPE>>& batch_query_ids,
      ID_TYPE num_batches,
      ID_TYPE num_error_quantiles,
      const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);
  
  ID_TYPE get_node_size_threshold() const {
    return node_size_threshold_;
  }

  // 新增：保存所有filter的预测alpha值到文件的方法声明
  RESPONSE save_predicted_alphas(const std::string& filepath);

  // 新增：模拟完整dstree搜索过程，重新计算准确的recall（接受root参数）
  RESPONSE simulate_full_search_for_recall(std::shared_ptr<dstree::Node> root);

 private:
 //QYL
  // std::vector<dstree::Answers>& train_answers_; // 新增变量

  RESPONSE trial_collect_mthread();
  RESPONSE evaluate();

  RESPONSE measure_cpu();
  RESPONSE measure_gpu();

  std::reference_wrapper<Config> config_;

  double_t cpu_ms_per_series_;
  ID_TYPE node_size_threshold_;

  std::vector<MODEL_SETTING> candidate_model_settings_;
  VALUE_TYPE available_gpu_memory_mb_;

  std::vector<FilterInfo> filter_infos_;

  bool is_recall_calculated_;
  std::vector<ERROR_TYPE> validation_recalls_;
  // 新增：存储每个批次的召回率
  std::vector<std::vector<ERROR_TYPE>> batch_validation_recalls_;
  // 新增：存储模拟完整搜索的每个批次召回率
  std::vector<std::vector<ERROR_TYPE>> batch_validation_recalls_simulated_;
  ERROR_TYPE min_validation_recall_;

  std::vector<ID_TYPE> filter_ids_;
  std::vector<VALUE_TYPE> gains_matrix_;
  std::vector<VALUE_TYPE> mem_matrix_;

  // 辅助函数：保存批次召回率结果到CSV文件
  RESPONSE save_batch_recall_results(
    const std::vector<std::vector<ID_TYPE>>& batch_query_ids,
    ID_TYPE num_batches,
    ID_TYPE num_error_quantiles,
    const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& query_knn_nodes);

  // 新增：模拟单个查询的dstree搜索过程
  ID_TYPE simulate_dstree_search_for_query(
      ID_TYPE query_id, 
      ID_TYPE batch_i, 
      ID_TYPE error_i,
      const std::unordered_map<ID_TYPE, size_t>& node_id_to_index,
      std::shared_ptr<dstree::Node> root);

  // TODO memory is continuous instead of discrete
//  std::unique_ptr<VALUE_TYPE> total_gain_matrix_;
//  std::unique_ptr<bool> path_matrix_;
  std::vector<std::vector<std::tuple<VALUE_TYPE, VALUE_TYPE>>> total_gain_matrix_;
  std::vector<std::vector<ID_TYPE>> path_matrix_;
};

} // namespace dstree
} // namespace upcite

#endif //DSTREE_SRC_EXEC_FILTER_ALLOCATOR_H_
