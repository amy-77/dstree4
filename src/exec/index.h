//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#ifndef DSTREE_INDEX_H
#define DSTREE_INDEX_H

#include <memory>
#include <vector>
#include <stack>
#include <unordered_map>

#include <torch/torch.h>

#include "global.h"
#include "config.h"
#include "buffer.h"
#include "node.h"
#include "filter.h"
#include "filter_allocator.h"
#include "navigator.h"
namespace upcite {
namespace dstree {

using NODE_DISTNCE = std::tuple<std::reference_wrapper<dstree::Node>, VALUE_TYPE>;

class CompareDecrNodeDist {
 public:
  bool operator()(const NODE_DISTNCE &a, const NODE_DISTNCE &b) {
    return std::get<1>(a) > std::get<1>(b);
  }
};

static bool compDecrProb(std::tuple<ID_TYPE, VALUE_TYPE> &a, std::tuple<ID_TYPE, VALUE_TYPE> &b) {
  return std::get<1>(a) > std::get<1>(b);
}



// 定义一个结构体来存储查询结果信息
struct QueryPredictionRecord {
  ID_TYPE query_id;                // 查询ID
  VALUE_TYPE true_nn_distance;     // 真实最近邻距离
  VALUE_TYPE predicted_distance;   // 预测距离
  VALUE_TYPE prediction_error;     // 预测误差 (predicted - true)
  VALUE_TYPE bsf_distance;         // 最小bsf距离
  ID_TYPE filter_id;               // 过滤器ID (即节点ID)
  ID_TYPE series_id;               // 最近邻序列ID
};

class Index {
 public:
  explicit Index(Config &config);
  ~Index();

  RESPONSE build();

  RESPONSE dump() const;
  RESPONSE load();
  RESPONSE load_enhanced();

  RESPONSE search(bool is_profile=false);
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr = nullptr,
                 dstree::Answers *results = nullptr);
  // RESPONSE search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
  //                dstree::Answers *results,
  //                ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter_total,
  //                ID_TYPE &nfpruned_node_counter_calib, ID_TYPE &nfpruned_series_counter_calib,
  //                ID_TYPE &nfpruned_node_counter_lb, ID_TYPE &nfpruned_series_counter_lb);
  RESPONSE search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                 dstree::Answers *results,
                 ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter_total,
                 ID_TYPE &nfpruned_node_counter_calib, ID_TYPE &nfpruned_series_counter_calib,
                 ID_TYPE &nfpruned_node_counter_lb, ID_TYPE &nfpruned_series_counter_lb,
                 torch::Tensor &filter_query_tensor);
  RESPONSE search_navigated(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr = nullptr);
 
  std::vector<upcite::dstree::QueryPredictionRecord> search_with_prediction_error(
    ID_TYPE query_id, 
    VALUE_TYPE *query_ptr, 
    VALUE_TYPE *sketch_ptr = nullptr);

  RESPONSE collect_all_query_prediction_data(const std::string& output_filepath);
  
  void store_bsf(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE bsf);
  VALUE_TYPE get_bsf(ID_TYPE query_id, ID_TYPE node_id);
  //  新增 Getter 方法
  // const std::vector<Answers>& get_train_answers() const {
  //   return train_answers_;
  // }
  // std::vector<Answers> train_answers_;
  
  // 新增获取所有叶子节点ID的方法
  std::vector<ID_TYPE> get_all_leaf_ids() const; // 注意 const 修饰符
  void calculate_recall();
  void calculate_batch_recall();
  // 新增get_query_knn_nodes， 用于allocate recall
  const std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>>& get_query_knn_nodes() const {
    return query_knn_nodes_;
  }
  ID_TYPE get_leaf_count() const {
    return nleaf_; 
  } 
  
  // 获取激活的过滤器数量
  int get_active_filter_count() const;

  // 新增：保存和加载ground truth结果
  RESPONSE save_ground_truth_results(const std::string& filepath, VALUE_TYPE* query_buffer) const;
  RESPONSE load_ground_truth_results(const std::string& filepath, VALUE_TYPE* query_buffer = nullptr);

  // Debug helper: print each query's KNN nodes and the corresponding min-bsf values
  void debug_print_knn_bsf();
  // Debug helper: print distance vectors for a specific filter
  void debug_print_filter_vectors(ID_TYPE filter_id);

  void log_pruning_to_csv(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE pred_raw_distance, 
                          VALUE_TYPE calib_error, VALUE_TYPE predicted_nn_distance, 
                          VALUE_TYPE minbsf, VALUE_TYPE true_nn_distance);  
  // Log all test data for further analysis
  void log_all_test_to_csv(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE pred_raw_distance,
                           VALUE_TYPE true_nn_distance, VALUE_TYPE predicted_nn_distance,
                           VALUE_TYPE calib_error, VALUE_TYPE true_error, VALUE_TYPE minbsf);
  
  // 从文件加载filter预测的error值
  RESPONSE load_filter_errors_from_file(const std::string& error_file_path);

 private:

  // 用于记录错误剪枝信息的文件流
  static std::ofstream wrong_pruning_file_;
  // 标记文件是否已初始化
  static bool wrong_pruning_file_initialized_;
  
  void init_wrong_pruning_file();
  // 递归计算激活的过滤器数量
  void count_active_filters(const Node& node, int& count) const;

  // QYL 在Index类中添加成员变量来存储ground truth和实际结果
  std::vector<std::shared_ptr<dstree::Answers>> ground_truth_answers_;
  std::vector<std::shared_ptr<dstree::Answers>> actual_answers_;
  std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, VALUE_TYPE>> query_node_bsf_map_;

  RESPONSE insert(ID_TYPE batch_series_id);

  RESPONSE train(bool is_retrain = false);

  RESPONSE profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                  dstree::Answers *results,
                  ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter_total);

  // initialize filter's member variables except the model
  RESPONSE filter_initialize(dstree::Node &node, ID_TYPE *filter_id);
  // to retrain
  RESPONSE filter_deactivate(dstree::Node &node);

  RESPONSE filter_collect();
  RESPONSE filter_collect_mthread();
  // 新增：基于filter_collect的多线程版本
  // RESPONSE filter_collect_mt();

  // assign model settings to filters and initialize their model variable
  RESPONSE filter_allocate(bool to_assign = true, bool reassign = false);

  RESPONSE filter_train();
  RESPONSE filter_train_mthread();
  //QYL: 存储每个query的KNN结果
  std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>> query_knn_nodes_;
  std::unordered_map<ID_TYPE, std::unordered_map<ID_TYPE, ID_TYPE>> query_knn_nodes_single_;

  std::reference_wrapper<Config> config_;

  std::unique_ptr<BufferManager> buffer_manager_;

  std::unique_ptr<Node> root_;
  ID_TYPE nnode_, nleaf_;

  std::unique_ptr<Allocator> allocator_;

  VALUE_TYPE *filter_train_query_ptr_;
  torch::Tensor filter_train_query_tsr_;
  torch::Tensor filter_query_tsr_;
  std::unique_ptr<torch::Device> device_;
  std::stack<std::reference_wrapper<Filter>> filter_cache_;
  // Map filter IDs to filter references for easy lookup
  std::unordered_map<ID_TYPE, Filter*> filter_id_to_filter_;

  std::vector<Answers> train_answers_;
  std::unique_ptr<Navigator> navigator_;
  std::vector<std::reference_wrapper<Node>> leaf_nodes_;
};

}
}

#endif //DSTREE_INDEX_H
