//
// Created by Qitong Wang on 2022/10/6.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include "index.h"

#include <tuple>
#include <memory>
#include <random>
#include <algorithm>
#include <immintrin.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <omp.h>

#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>
#include <functional> // 需要包含以使用 std::reference_wrapper
#include <string>  // 添加string头文件以支持std::string

#include "vec.h"
#include "eapca.h"
#include "answer.h"
#include "query_synthesizer.h"

namespace fs = boost::filesystem;

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;

// 静态成员变量定义
std::ofstream dstree::Index::wrong_pruning_file_;
bool dstree::Index::wrong_pruning_file_initialized_ = false;

dstree::Index::Index(Config &config) : config_(config),
                                       nnode_(0),
                                       nleaf_(0),
                                       filter_train_query_ptr_(nullptr),
                                       allocator_(nullptr),
                                       navigator_(nullptr)
{
  buffer_manager_ = std::make_unique<dstree::BufferManager>(config_);

  root_ = std::make_unique<dstree::Node>(config_, *buffer_manager_, 0, nnode_);
  nnode_ += 1, nleaf_ += 1;

  if (config_.get().filter_infer_is_gpu_){
    // TODO support multiple devices
    device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
  } else {
    device_ = std::make_unique<torch::Device>(torch::kCPU);
  }

  if (config_.get().require_neurofilter_){
    allocator_ = std::make_unique<dstree::Allocator>(config_);
  }
}



dstree::Index::~Index(){
  if (filter_train_query_ptr_ != nullptr)
  {
    std::free(filter_train_query_ptr_);
    filter_train_query_ptr_ = nullptr;
  }
    // 在这里添加关闭文件的代码
  if (wrong_pruning_file_initialized_ && wrong_pruning_file_.is_open()) {
    wrong_pruning_file_.close();
  }
}




// 构建树索引，这里insert函数就是构建index的关键流程，把数据插入到树中
// 构建好index以后, 调用train函数训练filter (train函数很长，包含了collect training数据，训练filter等)
RESPONSE dstree::Index::build(){
  // 记录总构建时间开始
  auto total_build_start_time = std::chrono::high_resolution_clock::now();
  printf("==========开始构建索引==========\n");
  spdlog::info("==========开始构建索引==========");
  
  //-------------------1. build index-------------------
  // 记录数据加载和插入时间开始
  auto data_insertion_start_time = std::chrono::high_resolution_clock::now();
  printf("----------开始数据加载和插入阶段----------\n");
  spdlog::info("----------开始数据加载和插入阶段----------");
  
  ID_TYPE total_series_inserted = 0;
  ID_TYPE batch_count = 0;
  
  while (buffer_manager_->load_batch() == SUCCESS){
    batch_count++;
    ID_TYPE current_batch_size = buffer_manager_->load_buffer_size();
    printf("---------------[DEBUG] Loaded batch %ld of size: %ld\n", (long)batch_count, (long)current_batch_size);
    
    for (ID_TYPE series_id = 0; series_id < current_batch_size; ++series_id){
      // 将当前数据点（由 series_id 标识）插入到树中。
      //  printf("[DEBUG] Inserting series_id: %d\n", series_id);
      insert(series_id);
      total_series_inserted++;
    }
    // printf("[DEBUG] Inserting series_id finished \n");
    if (config_.get().on_disk_){
      buffer_manager_->flush();
    }
  }
  
  // 计算数据加载和插入时间
  auto data_insertion_end_time = std::chrono::high_resolution_clock::now();
  auto data_insertion_duration = std::chrono::duration_cast<std::chrono::milliseconds>(data_insertion_end_time - data_insertion_start_time);
  
  printf("==========[DEBUG] 数据插入完成统计信息 ==========\n");
  printf("总共加载 %ld 个批次\n", (long)batch_count);
  printf("总共插入 %ld 个时间序列\n", (long)total_series_inserted);
  printf("叶子节点数量: %ld\n", (long)nleaf_);
  printf("数据加载和插入耗时: %ld 毫秒 (%.3f 秒)\n", data_insertion_duration.count(), data_insertion_duration.count() / 1000.0);
  
  spdlog::info("==========[DEBUG] 数据插入完成统计信息 ==========");
  spdlog::info("总共加载 {} 个批次", batch_count);
  spdlog::info("总共插入 {} 个时间序列", total_series_inserted);
  spdlog::info("叶子节点数量: {}", nleaf_);
  spdlog::info("数据加载和插入耗时: {} 毫秒 ({:.3f} 秒)", data_insertion_duration.count(), data_insertion_duration.count() / 1000.0);

  if (!buffer_manager_->is_fully_loaded()){ // 检查是否所有数据都已成功加载
    printf("-------[ERROR] Failed to fully load data.\n");
    return FAILURE;
  }
  // leaf_min_heap_: 这是一个优先队列，存储lb<minBSF的节点
  // leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
  //     CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));

  //------------------2. train filter 训练索引中的过滤器----------------
  // navigator 开头的参数都不用管; 这个是学一个模型用来改变叶节点的访问顺序的
  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_){
    //**********************   train filter  ********************* */
    printf("----------[DEBUG] 开始过滤器训练阶段...\n");
    spdlog::info("----------[DEBUG] 开始过滤器训练阶段...");
    
    auto filter_training_start_time = std::chrono::high_resolution_clock::now();
    train();
    auto filter_training_end_time = std::chrono::high_resolution_clock::now();
    auto filter_training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_training_end_time - filter_training_start_time);
    
    printf("过滤器训练耗时: %ld 毫秒 (%.3f 秒)\n", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
    spdlog::info("过滤器训练耗时: {} 毫秒 ({:.3f} 秒)", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
    
    // QYL: 传递 Index 的 train_answers_ 给Allocator
    // allocator_ = std::make_unique<dstree::Allocator>(train_answers_);
  }
  
  // 计算总构建时间
  auto total_build_end_time = std::chrono::high_resolution_clock::now();
  auto total_build_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_build_end_time - total_build_start_time);
  
  printf("==========索引构建完成==========\n");
  printf("总构建时间: %ld 毫秒 (%.3f 秒)\n", total_build_duration.count(), total_build_duration.count() / 1000.0);
  printf("[DEBUG] Index build completed successfully.\n");
  
  spdlog::info("==========索引构建完成==========");
  spdlog::info("总构建时间: {} 毫秒 ({:.3f} 秒)", total_build_duration.count(), total_build_duration.count() / 1000.0);
  spdlog::info("[DEBUG] Index build completed successfully.");
  
  return SUCCESS;
}



// 在 Index 类中添加
std::vector<ID_TYPE> upcite::dstree::Index::get_all_leaf_ids() const{
  // std::vector<std::reference_wrapper<Node>> leaves;
  std::vector<std::reference_wrapper<upcite::dstree::Node>> leaves; // 明确使用你的 Node 类
  root_->enqueue_leaf(leaves); // 收集所有叶子节点
  std::vector<ID_TYPE> leaf_ids;
  for (const auto &leaf_ref : leaves){
    leaf_ids.push_back(leaf_ref.get().get_id());
  }
  return leaf_ids;
}



//
RESPONSE dstree::Index::insert(ID_TYPE batch_series_id){
  if (config_.get().is_sketch_provided_){
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_sketch_ptr(batch_series_id),
        config_.get().sketch_length_,
        config_.get().vertical_split_nsubsegment_)));
  } else {
    buffer_manager_->emplace_series_eapca(std::move(std::make_unique<dstree::EAPCA>(
        buffer_manager_->get_series_ptr(batch_series_id),
        config_.get().series_length_,
        config_.get().vertical_split_nsubsegment_)));
  }
  dstree::EAPCA &series_eapca = buffer_manager_->get_series_eapca(batch_series_id);
  std::reference_wrapper<dstree::Node> target_node = std::ref(*root_);

  while (!target_node.get().is_leaf()){
    target_node = target_node.get().route(series_eapca, true);
  }
  if (target_node.get().is_full()){
    target_node.get().split(*buffer_manager_, nnode_);
    nnode_ += config_.get().node_nchild_, nleaf_ += config_.get().node_nchild_ - 1;
    target_node = target_node.get().route(series_eapca, true);
  }
  return target_node.get().insert(batch_series_id, series_eapca);
}



// 递归函数，给所有叶子节点插入初始filter，并且所有filter存入filter_cache中
RESPONSE dstree::Index::filter_initialize(dstree::Node &node,
                                          ID_TYPE *filter_id){
  if (!filter_id) {
    fprintf(stderr, "[ERROR] filter_initialize: filter_id pointer is null\n");
    return FAILURE;
  }

  try {
    if (node.is_leaf()) {
      // 为叶节点添加过滤器
      if (!filter_train_query_tsr_.defined()) {
        fprintf(stderr, "[ERROR] filter_initialize: filter_train_query_tsr_ is not defined\n");
        return FAILURE;
      }
      
      // 使用节点ID而不是顺序ID作为过滤器ID，更好地支持故障恢复
      ID_TYPE node_id = node.get_id();
      RESPONSE result = node.add_filter(node_id, filter_train_query_tsr_);
      
      if (result != SUCCESS) {
        fprintf(stderr, "[ERROR] Failed to add filter to node ID: %ld\n", (long)node_id);
        return FAILURE;
      }
      
      // 检查是否成功获取过滤器引用
      try {
        auto filter_ref = node.get_filter();
        filter_cache_.push(filter_ref);
        
        // 追踪过滤器ID到过滤器的映射（方便后续查找）
        filter_id_to_filter_[node_id] = &filter_ref.get();
        
        // printf("[INFO] Successfully initialized filter for node ID: %ld\n", (long)node_id);
      } catch (const std::exception& e) {
        fprintf(stderr, "[ERROR] Failed to get filter reference from node ID: %ld, error: %s\n", 
               (long)node_id, e.what());
        return FAILURE;
      }
      
      (*filter_id)++;
    } else {
      // 递归处理子节点
      bool child_success = true;
      for (auto child_node : node) {
        RESPONSE child_result = filter_initialize(child_node, filter_id);
        if (child_result != SUCCESS) {
          fprintf(stderr, "[WARNING] Failed to initialize filter for a child of node ID: %ld\n", 
                 (long)node.get_id());
          child_success = false;
        }
      }
      
      if (!child_success) {
        return FAILURE;
      }
    }
    
    return SUCCESS;
  } catch (const std::exception& e) {
    fprintf(stderr, "[ERROR] Exception in filter_initialize for node ID: %ld, error: %s\n", 
           (long)node.get_id(), e.what());
    return FAILURE;
  }
}

RESPONSE dstree::Index::filter_deactivate(dstree::Node &node){
  if (node.is_leaf()){
    if (node.has_active_filter()){
      node.deactivate_filter();
    }
  } else {
    for (auto child_node : node)
    {
      filter_deactivate(child_node);
    }
  }

  return SUCCESS;
}



// 通过遍历树结构，计算查询序列与树中节点的局部最近邻距离，并更新全局最佳距离（BSF）
RESPONSE dstree::Index::filter_collect(){
  // printf("\n ------------------ 进入filter_collect --------------\n");
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), sizeof(VALUE_TYPE) * 8));
  if (!m256_fetch_cache){
    fprintf(stderr, "[ERROR] Failed to allocate aligned memory for m256_fetch_cache\n");
    return FAILURE;
  }
  // 这里收集的数据是global query所有的global_1nn_distances和global_bsf_distances (针对每个filter)
  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id){
    const VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;
    ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

    // 初始化answer
    auto answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
    if (!answer) {
      fprintf(stderr, "[ERROR] Failed to create answer object\n");
      return FAILURE;
    }
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
    //----------- 近似搜索：第一部分：找到最近的叶子节点并计算局部最近邻距离 -----------------
    // step1: 找到和查询序列series_ptr最近的叶子节点
    // printf("=========进入第一部分: 近似搜索=========\n");
    // spdlog::info("=========进入第一部分: 近似搜索=========\n");
    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }
    // step2: 获取当前query(series_ptr)到叶节点resident_node下所有时间序列的最近local距离
    auto minbsf = answer->get_bsf();
    VALUE_TYPE local_nn_distance = resident_node.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
    if (std::isnan(local_nn_distance) || local_nn_distance < 0) {
      fprintf(stderr, "Invalid distance: %.3f\n", local_nn_distance);
      continue;
    }
    // printf("query %ld node %ld min_bsf = %.3f\n", static_cast<long>(query_id), static_cast<long>(resident_node.get().get_id()), minbsf);
    // spdlog::info("query {:d} node {:d} min_bsf = {:.3f}", query_id, resident_node.get().get_id(), minbsf);
    // 存储global query到当前叶节点的最近距离 在 global_nn_distance 和 global_bsf_distance中
    resident_node.get().push_global_example(minbsf, local_nn_distance, 0); // answer->get_bsf()应该存访问该节点之前的全局最近邻，
    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    // 每访问一个叶子节点，用当前节点的局部1NN距离来决定是否更新bsf_distances(minBSF)
    if (answer->is_bsf(local_nn_distance)) {
      // spdlog::info("filter query {:d} update bsf {:.3f} after node {:d} series {:d}", query_id, local_nn_distance, visited_node_counter, visited_series_counter);
      answer->push_bsf(local_nn_distance, resident_node.get().get_id());
    }
    // 使用本地优先队列，避免线程冲突
    std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
    local_leaf_min_heap.push(std::make_tuple(std::ref(*root_), 0));


    //------------------------ 第二部分：遍历其他叶子节点并更新全局最佳距离 --------------------
    // 创建优先队列，利用lb，使用优先队列（local_leaf_min_heap）遍历其他叶子节点，
    // printf("=========进入第二部分: 精确搜索=========\n");
    // spdlog::info("=========进入第二部分: 精确搜索=========\n");
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!local_leaf_min_heap.empty()){
      if (local_leaf_min_heap.empty()){
        printf("警告：堆为空但尝试访问元素\n");
        break;
      }

      // 每次从队列取出下界距离最小的节点进行处理
      std::tie(node_to_visit, node2visit_lbdistance) = local_leaf_min_heap.top();
      local_leaf_min_heap.pop();

      if (node_to_visit.get().is_leaf()) {
        //确认不是已经访问过的叶子节点
        if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
          auto minbsf = answer->get_bsf();
          local_nn_distance = node_to_visit.get().search(series_ptr, query_id, m256_fetch_cache, answer.get());
          //！！！把global query到当前叶节点的最近距离 在 global_lnn_distances 和 global_bsf_distances中
          // spdlog::info("query {:d} node {:d} min_bsf = {:.3f}", query_id, node_to_visit.get().get_id(), minbsf);
          // printf("query %ld node %ld min_bsf = %.3f\n", static_cast<long>(query_id), static_cast<long>(node_to_visit.get().get_id()), minbsf);
          node_to_visit.get().push_global_example(minbsf, local_nn_distance, node2visit_lbdistance);
          
          // printf("minbsf: %f, local_nn_distance: %f:\n", minbsf, local_nn_distance);
          visited_node_counter += 1;
          visited_series_counter += node_to_visit.get().get_size();
          if (answer->is_bsf(local_nn_distance)) {
            answer->push_bsf(local_nn_distance, node_to_visit.get().get_id());
          }
        }

      } else {
        // printf("-----search child node %d 's knn:  -----\n", node_to_visit.get().get_id());
        for (auto child_node : node_to_visit.get()) {
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(series_ptr);
          local_leaf_min_heap.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }

    }

    // 使用 get_current_topk() 拷贝结果，既用于统计也用于调试输出
    auto topk_answers = answer->get_current_topk();
    // for (size_t i = 0; i < topk_answers.size(); i++) {
    //   const auto& rec = topk_answers[i];
    //   printf("   query id %ld, nn_dist: %.6f, node_id: %ld, global_offset: %ld\n", 
    //         static_cast<long>(query_id), rec.nn_dist_, static_cast<long>(rec.node_id_), static_cast<long>(rec.global_offset_));
    // }

    // 更新 query_knn_nodes_ 统计表
    query_knn_nodes_[query_id].clear();
    for (const auto& rec : topk_answers) {
      if (rec.node_id_ > 0) {
        query_knn_nodes_[query_id][rec.node_id_]++;
      }
    }

    // 打印 Top-K 结果（从最近到最远）
    // spdlog::info("---- Query {} final KNN ----", query_id);
    // // printf("---- Query %ld final KNN ----\n", static_cast<long>(query_id));
    // for (const auto& rec : topk_answers) {
    //   spdlog::info("   node {}  dist {:.3f}", rec.node_id_, rec.nn_dist_);
    //   // printf("   node %ld  dist %.3f\n", static_cast<long>(rec.node_id_), rec.nn_dist_);
    // }

    // 打印最终的统计结果（调试用）
    // printf("---- query_knn_nodes_[%d] 统计结果 ----\n", query_id);
    // for (const auto& [node_id, count] : query_knn_nodes_[query_id]) {
    //     printf("query_id=%d, node_id=%d, count=%d\n", query_id, node_id, count);
    // }

    // 这个函数存储了global_1nn_distances和global_bsf_distances (针对每个filter),
    // 这里面的距离都是平方距离
    // 打印global_lnn_distances和global_bsf_distances距离
    // printf("global_lnn_distances: %f\n", global_lnn_distances_[query_id]);
    // printf("global_bsf_distances: %f\n", global_bsf_distances_[query_id]);
  }

  // Call the debug print function here
  // debug_print_knn_bsf();
  // Print global distance vectors for ALL filters (for debugging)
  // spdlog::info("========= 在index.cc中打印所有filter的距离向量 ========");
  // for (const auto &kv : filter_id_to_filter_) {
  //   ID_TYPE fid = kv.first;
  //   auto *fptr = kv.second;
  //   if (fptr == nullptr) {
  //     printf("[Index] Filter pointer for id %ld is null.\n", static_cast<long>(fid));
  //     continue;
  //   }
  //   spdlog::info("\n++++ Filter {} ++++", static_cast<long>(fid));
  //   fptr->debug_print_global_vectors();
  // }

  // 使用后确保释放内存
  std::free(m256_fetch_cache);
  return SUCCESS;
}



// ===============================
// Debug helper: print distance vectors for a specific filter
// ===============================
void dstree::Index::debug_print_filter_vectors(ID_TYPE filter_id) {
  auto it = filter_id_to_filter_.find(filter_id);
  if (it == filter_id_to_filter_.end()) {
    printf("[Index] Filter id %ld not found in filter_id_to_filter_ map.\n", static_cast<long>(filter_id));
    return;
  }
  if (it->second == nullptr) {
    printf("[Index] Filter pointer for id %ld is null.\n", static_cast<long>(filter_id));
    return;
  }
  it->second->debug_print_global_vectors();
}




// ---------- 调试函数：打印 query_knn_nodes_ + minbsf ----------
void dstree::Index::debug_print_knn_bsf() {
  // 1. 先把 node_id → Node* 建个快查表
  std::unordered_map<ID_TYPE, dstree::Node*> node_map;
  std::stack<std::reference_wrapper<dstree::Node>> stk;
  stk.push(std::ref(*root_));
  while (!stk.empty()) {
    auto n = stk.top(); stk.pop();
    node_map[n.get().get_id()] = &n.get();
    if (!n.get().is_leaf()) {
      for (auto child : n.get()) stk.push(child);
    }
  }

  printf("\n================ Debug: query-knn-minbsf =================\n");
  spdlog::info("================ Debug: query-knn-minbsf =================\n");
  for (const auto& [qid, node_cnt_map] : query_knn_nodes_) {
    printf("Query %ld:\n", (long)qid);
    spdlog::info("Query {}", (long)qid);
    for (const auto& [nid, cnt] : node_cnt_map) {
      auto it = node_map.find(nid);
      if (it == node_map.end()) {
        printf("  Node %ld NOT found in tree!\n", (long)nid);
        spdlog::info("  Node {} NOT found in tree!", (long)nid);
        continue;
      }
      dstree::Node* node_ptr = it->second;
      if (!node_ptr->has_filter()) {
        printf("  Node %ld  (count=%d)  -- no filter\n", (long)nid, (int)cnt);
        spdlog::info("  Node {}  (count={})  -- no filter", (long)nid, (int)cnt);
        continue;
      }
      VALUE_TYPE bsf = node_ptr->get_filter_bsf_distance(qid);
      printf("  Node %ld  (count=%d)  min_bsf = %.6f\n",
             (long)nid, (int)cnt, bsf);
      spdlog::info("  Node {}  (count={})  min_bsf = {:.6f}", (long)nid, (int)cnt, bsf);
    }
    printf("\n");
    spdlog::info("\n");
  }
  printf("==========================================================\n");
  spdlog::info("==========================================================\n");
}


struct SearchCache{
  SearchCache(ID_TYPE thread_id,
              VALUE_TYPE *m256_fetch_cache,
              dstree::Answers *answer,
              pthread_mutex_t *answer_mutex,
              std::reference_wrapper<std::priority_queue<dstree::NODE_DISTNCE,
                                                         std::vector<dstree::NODE_DISTNCE>,
                                                         dstree::CompareDecrNodeDist>>
                  leaf_min_heap,
              pthread_mutex_t *leaf_pq_mutex,
              ID_TYPE *visited_node_counter,
              ID_TYPE *visited_series_counter,
              pthread_mutex_t *log_mutex) : thread_id_(thread_id),
                                            query_id_(-1),
                                            query_series_ptr_(nullptr),
                                            m256_fetch_cache_(m256_fetch_cache),
                                            answer_(answer),
                                            answer_mutex_(answer_mutex),
                                            leaf_min_heap_(leaf_min_heap),
                                            leaf_pq_mutex_(leaf_pq_mutex),
                                            visited_node_counter_(visited_node_counter),
                                            visited_series_counter_(visited_series_counter),
                                            log_mutex_(log_mutex) {}

  ID_TYPE thread_id_;
  ID_TYPE query_id_;
  VALUE_TYPE *query_series_ptr_;
  VALUE_TYPE *m256_fetch_cache_;
  dstree::Answers *answer_;
  pthread_mutex_t *answer_mutex_;
  std::reference_wrapper<std::priority_queue<
      dstree::NODE_DISTNCE, std::vector<dstree::NODE_DISTNCE>, dstree::CompareDecrNodeDist>>
      leaf_min_heap_;
  pthread_mutex_t *leaf_pq_mutex_;
  ID_TYPE *visited_node_counter_;
  ID_TYPE *visited_series_counter_;
  pthread_mutex_t *log_mutex_;
};


// filter collect的search 函数内部
void search_thread_F(const SearchCache &search_cache){
  // aligned_alloc within thread might cause a "corrupted size vs. prev_size" glibc error
  // https://stackoverflow.com/questions/49628615/understanding-corrupted-size-vs-prev-size-glibc-error
  //  auto m256_fetch_cache = std::unique_ptr<VALUE_TYPE>(static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8)));
  // WARN undefined behaviour
  std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
  VALUE_TYPE node2visit_lbdistance;

  while (true){
    pthread_mutex_lock(search_cache.leaf_pq_mutex_);
    if (search_cache.leaf_min_heap_.get().empty()){
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
      break;
    } else {
      // 从队列顶部获取节点和下界距离
      std::tie(node_to_visit, node2visit_lbdistance) = search_cache.leaf_min_heap_.get().top();
      search_cache.leaf_min_heap_.get().pop();
      pthread_mutex_unlock(search_cache.leaf_pq_mutex_);
    }
    // 这个代码只统计了有过滤器的节点的最近邻，并且push_global_example
    if (node_to_visit.get().is_leaf()) {
    // if (node_to_visit.get().has_filter()){
      // 在搜索节点之前获取global_bsf，用于push_global_example的对比
      pthread_mutex_lock(search_cache.answer_mutex_);
      VALUE_TYPE global_bsf_before_search = search_cache.answer_->get_bsf();
      pthread_mutex_unlock(search_cache.answer_mutex_);

      // 执行多线程搜索，search_mt内部会获取最新的global_bsf
      VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
          search_cache.query_series_ptr_, search_cache.query_id_, *search_cache.answer_, search_cache.answer_mutex_);
      
      // 添加锁保护Filter数据修改操作
      pthread_mutex_lock(search_cache.answer_mutex_);  // 重用现有的mutex或创建新的filter_mutex
      node_to_visit.get().push_global_example(global_bsf_before_search, local_nn_distance, node2visit_lbdistance);
      pthread_mutex_unlock(search_cache.answer_mutex_);

      pthread_mutex_lock(search_cache.log_mutex_);
      *search_cache.visited_node_counter_ += 1;
      *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
      pthread_mutex_unlock(search_cache.log_mutex_);
    } //else {
    //   // 在搜索前获取当前global_bsf用于剪枝判断
    //   pthread_mutex_lock(search_cache.answer_mutex_);
    //   VALUE_TYPE current_global_bsf = search_cache.answer_->get_bsf();
    //   pthread_mutex_unlock(search_cache.answer_mutex_);
      
    //   if (node2visit_lbdistance <= current_global_bsf) {
    //     // 当前节点没有激活filter，并且lb无法减枝时，也需要访问当前节点，但是此时由于没有插入filter，所以不需要存储训练和校准数据。
    //     VALUE_TYPE local_nn_distance = node_to_visit.get().search_mt(
    //         search_cache.query_series_ptr_, search_cache.query_id_, *search_cache.answer_, search_cache.answer_mutex_);

    //     pthread_mutex_lock(search_cache.log_mutex_);
    //     *search_cache.visited_node_counter_ += 1;
    //     *search_cache.visited_series_counter_ += node_to_visit.get().get_size();
    //     pthread_mutex_unlock(search_cache.log_mutex_);
    //   }
    // }



    
  }
}


// 收集信息
RESPONSE dstree::Index::filter_collect_mthread(){
  auto *m256_fetch_cache = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), 8 * config_.get().filter_collect_nthread_));

  ID_TYPE visited_node_counter = 0;
  ID_TYPE visited_series_counter = 0;

  std::unique_ptr<Answers> answer = nullptr;
  if (config_.get().navigator_is_learned_){ // TODO
    assert(!config_.get().require_neurofilter_);
    answer = std::make_unique<dstree::Answers>(config_.get().navigator_train_k_nearest_neighbor_, -1);
  } else {
    answer = std::make_unique<dstree::Answers>(config_.get().n_nearest_neighbor_, -1);
  }

  std::unique_ptr<pthread_mutex_t> answer_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> leaf_pq_mutex = std::make_unique<pthread_mutex_t>();
  std::unique_ptr<pthread_mutex_t> log_mutex = std::make_unique<pthread_mutex_t>();

  pthread_mutex_init(answer_mutex.get(), nullptr);
  pthread_mutex_init(leaf_pq_mutex.get(), nullptr);
  pthread_mutex_init(log_mutex.get(), nullptr);

  std::vector<SearchCache> search_caches;
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;

  // 创建本地优先队列，替代成员变量
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;

  for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id){
    search_caches.emplace_back(thread_id,
                               m256_fetch_cache + 8 * thread_id,
                               answer.get(),
                               answer_mutex.get(),
                               std::ref(local_leaf_min_heap),
                               leaf_pq_mutex.get(),
                               &visited_node_counter,
                               &visited_series_counter,
                               log_mutex.get()); // 传递当前 query_id
  }


  // --------1. 近似搜索: 遍历每个query，找当前query最可能落在哪个叶子节点，就计算该叶子节点下的1NN-------
  
  // 添加进度跟踪
  auto start_time = std::chrono::high_resolution_clock::now();
  ID_TYPE total_queries = config_.get().filter_train_nexample_;
  
  for (ID_TYPE query_id = 0; query_id < config_.get().filter_train_nexample_; ++query_id){
    // 每处理500个query打印一次进度
    if (query_id > 0 && query_id % 500 == 0) {
      auto current_time = std::chrono::high_resolution_clock::now();
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time);
      double progress = (double)query_id / total_queries * 100.0;
      printf("处理进度: %ld/%ld (%.1f%%), 已用时: %ld 秒\n", 
             query_id, total_queries, progress, elapsed.count());
    }
    
    VALUE_TYPE *series_ptr = filter_train_query_ptr_ + config_.get().series_length_ * query_id;

    visited_node_counter = 0;
    visited_series_counter = 0;
    answer->reset(query_id);
    // 从根节点开始遍历树
    std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
    // 找到当前查询序列距离最近的叶子节点
    while (!resident_node.get().is_leaf()) {
      resident_node = resident_node.get().route(series_ptr);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id){
      search_caches[thread_id].query_id_ = query_id;
      search_caches[thread_id].query_series_ptr_ = series_ptr;
    }

    // 获取全局最佳距离，并在当前叶子节点中搜索
    VALUE_TYPE global_bsf_distance = answer->get_bsf();
    VALUE_TYPE local_nn_distance = resident_node.get().search_mt(series_ptr, query_id, std::ref(*answer.get()), answer_mutex.get());

    // QYL  这个位置要改！！！！！！！！！当前节点的局部最近邻会被存到Filter.h的global_lnn_distances_中,在搜索阶段根据recall计算oi(delta)时用到
    resident_node.get().push_global_example(global_bsf_distance, local_nn_distance, 0);

    visited_node_counter += 1;
    visited_series_counter += resident_node.get().get_size();

    // 遍历整棵树的所有叶子节点（除了resident_node），计算其下界距离并存入local_leaf_min_heap，为后续多线程搜索生成候选队列。
    assert(node_stack.empty() && local_leaf_min_heap.empty());
    node_stack.push(std::ref(*root_));

    while (!node_stack.empty()){
      std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
      node_stack.pop();

      if (node_to_visit.get().is_leaf()){
        if (node_to_visit.get().get_id() != resident_node.get().get_id()){
          local_leaf_min_heap.push(std::make_tuple(node_to_visit, node_to_visit.get().cal_lower_bound_EDsquare(series_ptr)));
        }
      } else {
        for (auto child_node : node_to_visit.get()){
          node_stack.push(child_node);
        }
      }
    }

    // 其实这里 才是开始精确搜索，当前查询会遍历所有lb无法减枝的叶子节点进行KNN搜索
    std::vector<std::thread> threads;
    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id){
      threads.emplace_back(search_thread_F, search_caches[thread_id]);
    }

    for (ID_TYPE thread_id = 0; thread_id < config_.get().filter_collect_nthread_; ++thread_id){
      threads[thread_id].join();
    }

    // 11111111111111111   收集每个查询的答案
    train_answers_.emplace_back(dstree::Answers(*answer));

    // 使用 get_current_topk() 拷贝结果，既用于统计也用于调试输出
    auto topk_answers = answer->get_current_topk();
    
    // 调试输出：打印每个query的Top-K结果
    // if (query_id < 5) {
    //   printf("=== Query %ld Top-K Results ===\n", query_id);
    //   for (size_t i = 0; i < topk_answers.size(); i++) {
    //     const auto& rec = topk_answers[i];
    //     printf("   query id %ld, nn_dist: %.6f, node_id: %ld, global_offset: %ld\n", 
    //           static_cast<long>(query_id), rec.nn_dist_, static_cast<long>(rec.node_id_), static_cast<long>(rec.global_offset_));
    //   }
    // }

    // 更新 query_knn_nodes_ 统计表
    query_knn_nodes_[query_id].clear();
    for (const auto& rec : topk_answers) {
      if (rec.node_id_ > 0) {
        query_knn_nodes_[query_id][rec.node_id_]++;
      }
    }

    // 打印最终的统计结果（调试用）
    // printf("---- query_knn_nodes_[%ld] 统计结果 ----\n", query_id);
    // for (const auto& [node_id, count] : query_knn_nodes_[query_id]) {
    //     printf("query_id=%ld, node_id=%ld, count=%ld\n", query_id, node_id, count);
    // }

    ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;
    if (config_.get().navigator_is_learned_){
      nnn_to_return = config_.get().navigator_train_k_nearest_neighbor_;
    }

    // while (!answer->empty()){
    //   auto answer_i = answer->pop_answer();
    //   if (answer_i.node_id_ > 0)
    //   {
    //     printf("query %ld nn %ld = %.3f, node %ld\n", query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
    //     spdlog::info("query {:d} nn {:d} = {:.3f}, node {:d}", query_id, nnn_to_return, answer_i.nn_dist_, answer_i.node_id_);
    //   } else {
    //     printf("query %ld nn %ld = %.3f, db_global_offset_ %ld\n", query_id, nnn_to_return, sqrt(answer_i.nn_dist_), answer_i.global_offset_);
    //     spdlog::info("query {} nn {} = {:.3f}, node {}, db_global_offset_ {}",
    //                  query_id, nnn_to_return, sqrt(answer_i.nn_dist_), answer_i.node_id_, answer_i.global_offset_);
    //   }
    //   nnn_to_return -= 1;
    // }
  }

  // 添加处理完成的总结信息
  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_elapsed = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  printf("查询处理完成! 总共处理: %ld 个查询, 总用时: %ld 秒\n", 
         total_queries, total_elapsed.count());

  // 在释放内存前，先销毁 mutex
  pthread_mutex_destroy(answer_mutex.get());
  pthread_mutex_destroy(leaf_pq_mutex.get());
  pthread_mutex_destroy(log_mutex.get());

  std::free(m256_fetch_cache);
  return SUCCESS;
}



// 遍历树结构，收集叶子节点的剪枝信息，并根据配置触发过滤器的动态分配。
RESPONSE dstree::Index::filter_allocate(bool to_assign, bool reassign){
  printf("[DEBUG] Entering dstree::Index::filter_allocate()\n");
  printf("[DEBUG] to_assign = %d, reassign = %d\n", to_assign, reassign);
  // 这里to_assign = 1, reassign = 0
  std::stack<std::reference_wrapper<dstree::Node>> node_cache;
  node_cache.push(std::ref(*root_));
  // printf("[DEBUG] Pushed root node to stack.\n");
  // 1. 树遍历与剪枝信息收集
  while (!node_cache.empty()){
    // printf("[DEBUG] Processing next node in stack...\n");
    std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
    node_cache.pop();
    /*
    遍历树结构中的所有节点，把所有的叶子节点都插入filter
    如果是叶子节点，记录减枝率并且把该节点的filter_info存到filter_infos_
    如果是内部节点，将其子节点推入node_cache
    */
    if (node_to_visit.get().is_leaf()){
      // printf("[DEBUG] Node is a leaf node.\n");
      FilterInfo filter_info(node_to_visit);
      // printf(" filter_id %d /n", filter_info.node_.get().get_id());
      // 利用lb减枝的比例, lb_pruned count/ lb_size
      filter_info.external_pruning_probability_ = node_to_visit.get().get_envelop_pruning_frequency();
      // printf("[DEBUG] Pushing filter info for leaf node ID: %d\n", node_to_visit.get().get_id());
      // printf("filter_info.external_pruning_probability_: %d\n", filter_info.external_pruning_probability_);
      allocator_->push_filter_info(filter_info);
    } else {
      // printf("[DEBUG] Node is an internal node. Pushing child nodes to stack.\n");
      for (auto child_node : node_to_visit.get()){ // 获取当前内部节点的子节点列表
        // printf("[DEBUG] Pushing child node with ID: %d\n", child_node.get().get_id());
        node_cache.push(child_node);
      }
    }
  }
  // 2. 过滤器分配决策
  //  这里是选择合适的叶子节点进行active激活
  if (to_assign){
    printf("[DEBUG] Calling allocator_->assign()...\n");
    // allocator_->assign()这个函数是选择合适(gain最大/节点size大)的叶子节点激活filter
    return allocator_->assign();
    printf("[DEBUG] finished allocator_->assign()...\n");
  }
  else if (reassign){
    printf("[DEBUG] Calling allocator_->reassign()...\n");
    return allocator_->reassign();
  }else{
    printf("[DEBUG] No assignment or reassignment required. Returning SUCCESS.\n");
    return SUCCESS;
  }
}



// QYL batch
RESPONSE dstree::Index::filter_train(){
  printf("开始训练过滤器，filter_cache_大小: %zu\n", filter_cache_.size());
  spdlog::info("开始训练过滤器，filter_cache_大小: {}", filter_cache_.size());
  // Added: Track processed filters to avoid duplicates in case of issues
  std::unordered_set<ID_TYPE> processed_filter_ids;
  size_t filter_count = filter_cache_.size();
  size_t active_filters_processed = 0;
  size_t inactive_filters_skipped = 0;
  
  // Safely process the filter cache
  while (!filter_cache_.empty()) {
    // Get the top filter reference
    std::reference_wrapper<Filter> filter = filter_cache_.top();
    filter_cache_.pop();
    // Get filter ID for tracking
    ID_TYPE filter_id = filter.get().get_id();
    // Skip already processed filters (shouldn't happen but added as safety)
    if (processed_filter_ids.find(filter_id) != processed_filter_ids.end()) {
      printf("警告: 过滤器ID: %ld 已经处理过，跳过\n", filter_id);
      continue;
    }
    // Process active filters
    if (filter.get().is_active()) {
      printf("正在训练节点ID: %ld 的过滤器\n", filter_id);
      
      try {
        RESPONSE result = filter.get().batch_train();
        if (result != SUCCESS) {
          printf("警告: 过滤器ID: %ld 训练失败\n", filter_id);
        } else {
          active_filters_processed++;
        }
      } catch (const std::exception& e) {
        printf("错误: 过滤器ID: %ld 训练时发生异常: %s\n", filter_id, e.what());
      }
    } else {
      inactive_filters_skipped++;
    }
    
    // Mark as processed
    processed_filter_ids.insert(filter_id);
  }
  
  // Summary report
  printf("过滤器训练完成: 总共 %zu 个, 处理了 %zu 个活跃过滤器, 跳过 %zu 个非活跃过滤器\n", 
         filter_count, active_filters_processed, inactive_filters_skipped);
  
  return SUCCESS;
}




struct TrainCache{
  TrainCache(ID_TYPE thread_id,
             at::cuda::CUDAStream stream,
             std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache,
             pthread_mutex_t *filter_cache_mutex) : thread_id_(thread_id),
                                                    stream_(stream),
                                                    filter_cache_(filter_cache),
                                                    filter_cache_mutex_(filter_cache_mutex) {}

  ~TrainCache() = default;
  ID_TYPE thread_id_;
  at::cuda::CUDAStream stream_;
  // TODO remove ref to filter; use node instead
  std::stack<std::reference_wrapper<dstree::Filter>> &filter_cache_;
  pthread_mutex_t *filter_cache_mutex_;
};




// This is the training function used by each thread
void train_thread_F(TrainCache &train_cache) {
  at::cuda::setCurrentCUDAStream(train_cache.stream_);
  at::cuda::CUDAStreamGuard guard(train_cache.stream_); // compiles with libtorch-gpu
  
  // Added: Track processed filters for this thread
  std::unordered_set<ID_TYPE> processed_filters;
  size_t processed_count = 0;
  size_t active_count = 0;

  while (true) {
    // 添加互斥锁
    pthread_mutex_lock(train_cache.filter_cache_mutex_);
    // filter_cache_是一个包含多个Filter的栈，而每个线程可以获取多个过滤器，直到栈为空为止
    if (train_cache.filter_cache_.empty()) {
      // 解锁
      pthread_mutex_unlock(train_cache.filter_cache_mutex_);
      break;
    } else {
      // 主要进入这个分支
      std::reference_wrapper<dstree::Filter> filter = train_cache.filter_cache_.top(); // 获取栈顶元素
      train_cache.filter_cache_.pop();                                                 // 移除栈顶元素
      pthread_mutex_unlock(train_cache.filter_cache_mutex_);                           // 解锁
      // Get filter ID
      ID_TYPE filter_id = filter.get().get_id();
      // Check if this filter was already processed by this thread (shouldn't happen, but for safety)
      if (processed_filters.find(filter_id) != processed_filters.end()) {
        printf("Thread %d: 警告 - 过滤器 %ld 已被处理过，跳过\n", 
               train_cache.thread_id_, static_cast<long>(filter_id));
        continue;
      }
      
      // 如果当前这个叶子节点的filter被激活了，则进行Mi的训练
      if (filter.get().is_active()) {
        // printf("Thread %d: 开始训练过滤器 ID: %ld\n", train_cache.thread_id_, static_cast<long>(filter_id));
        try {
          // 这里调用的filter.train和单线程调用的是一样的，里面包含了Conformal Prediction的过程
          auto train_start = std::chrono::high_resolution_clock::now();

          RESPONSE result = filter.get().batch_train();
          auto train_end = std::chrono::high_resolution_clock::now();
          auto train_duration = std::chrono::duration_cast<std::chrono::milliseconds>(train_end - train_start);
          printf("Thread %d: 过滤器 ID: %ld 训练耗时: %ld ms\n", 
                 train_cache.thread_id_, static_cast<long>(filter_id), train_duration.count());
          if (result == SUCCESS) {
            active_count++;
            // printf("Thread %d: 成功训练过滤器 ID: %ld\n", train_cache.thread_id_, static_cast<long>(filter_id));
          } else {
             printf("Thread %d: 警告 - 过滤器 ID: %ld 训练失败\n", train_cache.thread_id_, static_cast<long>(filter_id));
          }
        } catch (const std::exception& e) {
          printf("Thread %d: 错误 - 过滤器 ID: %ld 训练时发生异常: %s\n", 
                 train_cache.thread_id_, static_cast<long>(filter_id), e.what());
        }
      }
      
      // Track this filter as processed
      processed_filters.insert(filter_id);
      processed_count++;
    }
  }
  // printf("Thread %d: 完成任务，共处理 %zu 个过滤器，成功训练 %zu 个活跃过滤器\n", 
  //        train_cache.thread_id_, processed_count, active_count);
}




// 这个是多线程train filter的总函数，里面每个线程调用train_thread_F
RESPONSE dstree::Index::filter_train_mthread() {
  // 多线程训练需要GPU支持
  assert(config_.get().filter_train_is_gpu_);
  assert(torch::cuda::is_available());

  printf("\n------------ filter_train_mthread开始 -------------\n");
  printf("filter_train_nthread_: %ld, 激活的过滤器数量: %zu\n", 
         (long)config_.get().filter_train_nthread_, filter_cache_.size());
  
  if (filter_cache_.empty()) {
    printf("警告: filter_cache_ 为空，没有过滤器需要训练\n");
    return SUCCESS;
  }
  
  // 创建过滤器栈和互斥锁
  std::stack<std::reference_wrapper<dstree::Filter>> filters;
  std::unique_ptr<pthread_mutex_t> filter_stack_mutex = std::make_unique<pthread_mutex_t>();
  pthread_mutex_init(filter_stack_mutex.get(), nullptr);

  // 统计活跃过滤器数量
  ID_TYPE active_filter_count = 0;
  
  // 收集所有过滤器
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;
  node_stack.push(std::ref(*root_));

  while (!node_stack.empty()) {
    std::reference_wrapper<dstree::Node> node_to_visit = node_stack.top();
    node_stack.pop();

    if (node_to_visit.get().is_leaf()) {
      if (node_to_visit.get().has_filter()) {
        filters.push(node_to_visit.get().get_filter());

        if (node_to_visit.get().has_active_filter()) {
          active_filter_count++;
        }

      }
    } else {
      for (auto child_node : node_to_visit.get()) {
        node_stack.push(child_node);
      }
    }
  }

  printf("收集到 %zu 个过滤器，其中 %ld 个激活\n", filters.size(), (long)active_filter_count);
  
  if (filters.empty()) {
    printf("警告: 没有找到过滤器，训练终止\n");
    return SUCCESS;
  }

#ifdef DEBUG
  spdlog::debug("indexing filters.size = {:d}", filters.size());
#endif
  printf("debug:filters.size = %zu\n", filters.size());

  // 创建训练缓存
  std::vector<std::unique_ptr<TrainCache>> train_caches;
  
  // 确定线程数量不超过过滤器数量
  ID_TYPE num_threads = std::min(
      config_.get().filter_train_nthread_, 
      static_cast<ID_TYPE>(filters.size())
  );
  
  if (num_threads < config_.get().filter_train_nthread_) {
    printf("注意: 调整线程数量从 %ld 到 %ld 以匹配过滤器数量\n", 
           (long)config_.get().filter_train_nthread_, (long)num_threads);
  }

  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    at::cuda::CUDAStream new_stream = at::cuda::getStreamFromPool(false, config_.get().filter_device_id_);

    // printf("创建训练线程 %ld, CUDA流ID: %ld\n", 
    //        (long)thread_id, (long)static_cast<ID_TYPE>(new_stream.id()));

// #ifdef DEBUG
//     spdlog::info("train thread {:d} stream id = {:d}, query = {:d}, priority = {:d}",
//                  thread_id,
//                  static_cast<ID_TYPE>(new_stream.id()),
//                  static_cast<ID_TYPE>(new_stream.query()),
//                  static_cast<ID_TYPE>(new_stream.priority())); // compiles with libtorch-gpu
// #endif

    train_caches.emplace_back(std::make_unique<TrainCache>(thread_id,                  // 线程 ID
                                                           std::move(new_stream),      // 独占的 CUDA 流
                                                           std::ref(filters),          // 所有线程共享的过滤器栈
                                                           filter_stack_mutex.get())); // 共享的互斥锁
  }

  // 创建并启动训练线程
  std::vector<std::thread> threads;
  printf("启动 %ld 个训练线程...\n", (long)num_threads);
  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    //------------每个线程都调用一个train_thread_F，这个函数是filter_train_mthread的核心------------------
    // 每个线程通过 std::ref(*train_caches[thread_id]) 获取自己的 TrainCache 引用
    threads.emplace_back(train_thread_F, std::ref(*train_caches[thread_id]));
  }
  // 等待所有线程完成
  // printf("等待所有训练线程完成...\n");
  for (ID_TYPE thread_id = 0; thread_id < num_threads; ++thread_id) {
    threads[thread_id].join();
  }
  // printf("------------ filter_train_mthread完成 -------------\n");
  return SUCCESS;
}





//  这个是train filter的过程，这个函数被总函数 dstree::Index::build() 调用
RESPONSE dstree::Index::train(bool is_retrain){
  printf("-----进入dstree::Index::train()--------\n");
  // ---------------------- 1. 生成或加载训练数据  ----------------ßß------
  // 创建查询生成器，用于生成训练过滤器的查询数据
  // local query generation is called after collecting global results
  dstree::Synthesizer query_synthesizer(config_, nleaf_);
  // filter_query_filepath_ 实际输入了，是用来训练filter的查询数据的文件路径
  if (!fs::exists(config_.get().filter_query_filepath_)){
    // 不走下面这个分支，而是else分支，else分支不重新train filter
    printf("[DEBUG] Filter query file does not exist. Generating synthetic queries...\n");
    assert(!is_retrain); // not applicable to loaded filters

    if (!config_.get().filter_query_filepath_.empty()){
      spdlog::error("filter train query filepath {:s} does not exist", config_.get().filter_query_filepath_);
      return FAILURE;
    }

    ID_TYPE query_set_nbytes = -1;
    // 根据CPU/GPU配置选择生成查询数据的模式
    ////针对每个过滤器生成特定数量的查询
    if (config_.get().filter_num_synthetic_query_per_filter_ > 0){ 
      printf("index::train() 进入分支 filter_num_synthetic_query_per_filter_: %ld\n",
           config_.get().filter_num_synthetic_query_per_filter_);
      ID_TYPE num_synthetic_queries = root_->get_num_synthetic_queries(allocator_->get_node_size_threshold());

      query_set_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * num_synthetic_queries;
      filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_set_nbytes));

      ID_TYPE num_generated_queries = 0;
      root_->synthesize_query(filter_train_query_ptr_, num_generated_queries, allocator_->get_node_size_threshold());
      assert(num_generated_queries == num_synthetic_queries);

      config_.get().filter_train_nexample_ = num_synthetic_queries;

      // spdlog::info("filter generated {:d} synthetic train queries", config_.get().filter_train_nexample_);
      
    }  else if (config_.get().filter_train_num_global_example_ > 0) {
      
      
      // 实际使用： 生成固定总数的全局查询样本用于所有过滤器
      printf("\n----------1.index::train() 进入分支 filter_train_num_global_example_: %ld----------\n",
           config_.get().filter_train_num_global_example_);
      // 为全局查询(global queries)分配内存空间
      query_set_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().filter_train_num_global_example_;
      // filter_train_query_ptr_是一个指针，指向存储所有训练查询数据的内存块。
      filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_set_nbytes));
      // 搜索并收集符合阈值条件的叶子节点，存入query_synthesizer
      ID_TYPE leaf_size_threshold = config_.get().filter_default_node_size_threshold_;
      std::stack<std::reference_wrapper<dstree::Node>> node_cache;
      node_cache.push(std::ref(*root_));
      while (!node_cache.empty()){
        std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
        node_cache.pop();
        if (node_to_visit.get().is_leaf()){
          if (node_to_visit.get().get_size() >= leaf_size_threshold || config_.get().to_profile_filters_){
            query_synthesizer.push_node(node_to_visit);
          }
        }
        else{
          for (auto child_node : node_to_visit.get()){
            node_cache.push(child_node);
          }
        }
      }

      // 生成全局查询样本，这里没有计算距离 global_nn_distance，后续会进入filter_collect()函数存储全局查询到各节点的最近距离
      printf("\n---------- 2. 生成全局查询样本 ----------\n");
      RESPONSE return_code = query_synthesizer.generate_global_data(filter_train_query_ptr_);
      
      config_.get().filter_train_nexample_ = config_.get().filter_train_num_global_example_;
      if (return_code == FAILURE){
        spdlog::error("failed to generate global queries");
        return FAILURE;
      }
      // printf("filter generated %d synthetic global queries\n", config_.get().filter_train_nexample_);
      spdlog::info("filter generated {:d} synthetic global queries", config_.get().filter_train_nexample_);
      // local query generation is called after collecting global results


      // Save the global query data as a separate test dataset  
      std::string test_dataset_filename = "global_query_test_" + std::to_string(config_.get().filter_train_nexample_) + ".bin";
      if (!config_.get().test_dataset_filename_.empty()) {
        test_dataset_filename = config_.get().test_dataset_filename_;
      }
      std::string test_dataset_filepath = config_.get().index_dump_folderpath_ + test_dataset_filename;
      printf("[INFO] Saving global query data as test dataset: %s\n", test_dataset_filepath.c_str());
      std::ofstream test_dataset_fout(test_dataset_filepath, std::ios::binary);
      if (test_dataset_fout.good()) {
        test_dataset_fout.write(reinterpret_cast<char *>(filter_train_query_ptr_), query_set_nbytes);
        test_dataset_fout.close();
        printf("[INFO] Successfully saved %ld global queries (%ld dimensions each) to test dataset\n", 
               static_cast<long>(config_.get().filter_train_nexample_), 
               static_cast<long>(config_.get().series_length_));
        spdlog::info("Saved global query test dataset to {}", test_dataset_filepath);
      } else {
        printf("[ERROR] Failed to save global query test dataset to %s\n", test_dataset_filepath.c_str());
        spdlog::error("Failed to save global query test dataset to {}", test_dataset_filepath);
      }



    } else {
      spdlog::error("erroneous config for query generation");
      return FAILURE;
    }

    assert(query_set_nbytes > 0);
    std::string filter_query_filepath = config_.get().index_dump_folderpath_ + config_.get().filter_query_filename_;
    printf("[DEBUG] filter_query_filepath = %s\n", filter_query_filepath.c_str());
    std::ofstream query_fout(filter_query_filepath, std::ios::binary | std::ios_base::app);
    query_fout.write(reinterpret_cast<char *>(filter_train_query_ptr_), query_set_nbytes);
    query_fout.close();

    
  } else {

    // 目前没有用到
    // filter_query_filepath_不为空，此时不用生成合成数据，直接
    printf("\n-------- 1. index.train: Loading filter query data from file...\n");
    printf("-------  filter_query_filepath_ = %s\n", config_.get().filter_query_filepath_.c_str());

    auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().filter_train_nexample_;
    filter_train_query_ptr_ = static_cast<VALUE_TYPE *>(aligned_alloc(sizeof(__m256), query_nbytes));
    std::ifstream query_fin(config_.get().filter_query_filepath_, std::ios::in | std::ios::binary);

    if (!query_fin.good()) {
      spdlog::error("filter train query filepath {:s} cannot open", config_.get().filter_query_filepath_);
      printf("[ERROR] filter train query filepath %s cannot open\n", config_.get().filter_query_filepath_.c_str());
    }

    query_fin.read(reinterpret_cast<char *>(filter_train_query_ptr_), query_nbytes);
    if (query_fin.fail()) {
      spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().filter_query_filepath_);
      printf("[ERROR] cannot read %ld bytes from %s\n", query_nbytes, config_.get().filter_query_filepath_.c_str());
      std::free(filter_train_query_ptr_);
      filter_train_query_ptr_ = nullptr;
    }
  }


  // ---------------------- 2. 配置训练设备（CPU/GPU） ----------------------
  // support difference devices for training and inference
  // printf("[DEBUG] 2. Configuring training device...\n");
  if (config_.get().require_neurofilter_){
    if (config_.get().filter_train_is_gpu_){
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }
  else if (config_.get().navigator_is_learned_){
    if (config_.get().navigator_is_gpu_) {
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  // ---------------------- 3. 将查询数据转换为PyTorch张量 ----------------------
  // printf("[DEBUG] 3. Converting query data to PyTorch tensor...\n");
  filter_train_query_tsr_ = torch::from_blob(filter_train_query_ptr_,
                                             {config_.get().filter_train_nexample_, config_.get().series_length_},
                                             torch::TensorOptions().dtype(TORCH_VALUE_TYPE));
  filter_train_query_tsr_ = filter_train_query_tsr_.to(*device_);

  if (config_.get().require_neurofilter_) {
    if (is_retrain) {
      printf("[DEBUG] Deactivating filters for retraining...\n");
      filter_deactivate(*root_);
    } else {
      // initialize filters
      ID_TYPE filter_id = 0;
      // 给所有叶子节点插入初始filter，并且所有filter存入filter_cache中
      filter_initialize(*root_, &filter_id);
      spdlog::info("initialized {:d} filters", filter_id);
    }
  }

  if (!is_retrain){
    // printf("[DEBUG] Collecting filter training data...\n");
    // collect filter training data, i.e., the bsf distances, nn distances, low-bound distances
    // if (config_.get().filter_train_is_mthread_)
    if (config_.get().filter_collect_is_mthread_) {
      printf("\n---------- 5. multi-threaded filter collection.\n");
      auto start_time = std::chrono::high_resolution_clock::now();
      // 收集train filter之前需要的局部真实最近邻数据，这里就获得了每个query到每个节点下的local_nn_distance
      filter_collect_mthread();
      auto end_time = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
      printf("Multi-threaded filter collection completed in %ld ms\n", duration.count());
      spdlog::info("Multi-threaded filter collection completed in {} ms", duration.count());
    } else {

      printf("\n----------5. single-threaded filter collection.------\n");
      auto start_time_single = std::chrono::high_resolution_clock::now();
      filter_collect();
      auto end_time_single = std::chrono::high_resolution_clock::now();
      auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(end_time_single - start_time_single);
      printf("Single-threaded filter collection completed in %ld ms\n", duration_single.count());
      spdlog::info("Single-threaded filter collection completed in {} ms", duration_single.count());
    }

    //生成合成查询作为local query
    // printf("\n---------- 6. 生成合成查询作为local query ----------\n");
    if (config_.get().filter_train_num_local_example_ > 0) {
      // generate and *search* local queries
      printf("\n---------- 6. Generating local queries... ----------\n");
      RESPONSE return_code = query_synthesizer.generate_local_data();

      if (return_code == FAILURE) { 
        printf("filter_train_num_local_example_ > 0, failed to generate local queries\n");
        spdlog::error("failed to generate local queries");
        return FAILURE;
      }
      printf("filter_train_num_local_example_ > 0, filter generated synthetic local queries: %d\n",config_.get().filter_train_num_local_example_);
      spdlog::info("filter generated {:d} synthetic local queries", config_.get().filter_train_num_local_example_);
    }
  }

  // ---------------------- index.train.filter_allocate 训练filter ----------------------
  if (config_.get().require_neurofilter_) {
    // allocate filters among nodes (and activate them)
    printf("\n-------6. Allocating filters---------\n");
    // 给所有叶子节点插入filter，并选择合适的叶子节点进行激活
    filter_allocate(true);
    // train all filter model
    // 打印所有过滤器的local_lnn_distances_和global_lnn_distances_大小
    // printf("-----打印所有过滤器的训练数据统计-----\n");
    // spdlog::info("-----打印所有过滤器的训练数据统计-----");
    // 使用STL来复制过滤器ID值并打印统计信息
    // std::vector<ID_TYPE> filter_ids;
    // for (const auto& filter_pair : filter_id_to_filter_) {
    //   ID_TYPE filter_id = filter_pair.first;
    //   auto& filter = filter_pair.second;
    //   filter_ids.push_back(filter_id);
      
      // 获取该过滤器的全局和本地距离向量大小
      // size_t global_size = filter->get_global_lnn_distances_size();
      // size_t local_size = filter->get_local_lnn_distances_size();
       // printf("过滤器 ID: %d\n", filter_id);
      // spdlog::info("过滤器 ID: {}", filter_id);
      
      // 如果有数据，打印距离的均值和标准差
      // if (global_size > 0) {
        // auto [global_mean, global_std] = filter->get_global_lnn_mean_std();
        // printf("  全局距离 - 均值: %.2f,   标准差: %.2f\n", global_mean, global_std);
        // spdlog::info("  全局距离 - 均值: {:.2f},   标准差: {:.2f}", global_mean, global_std);
      // }
      
      // if (local_size > 0) {
        // auto [local_mean, local_std] = filter->get_filter_local_lnn_mean_std();
        // printf("  本地距离 - 均值: %.2f,   标准差: %.2f\n", local_mean, local_std);
        // spdlog::info("  本地距离 - 均值: {:.2f},   标准差: {:.2f}", local_mean, local_std);
      // }
    // }
    // spdlog::info("总共 {} 个过滤器", filter_ids.size());
    // spdlog::info("-----统计结束-----");

    printf("\n---------- 7. 训练filter ----------\n");
    auto filter_training_start_time = std::chrono::high_resolution_clock::now();
    if (config_.get().filter_train_is_mthread_){
      printf("[DEBUG] Using multi-threaded filter training, filter_train_is_mthread_ is: %d\n", config_.get().filter_train_is_mthread_);
      filter_train_mthread();
      auto filter_training_end_time = std::chrono::high_resolution_clock::now();
      auto filter_training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_training_end_time - filter_training_start_time);
      printf("多线程Filter训练时间: %ld 毫秒 (%.3f 秒)\n", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
      spdlog::info("多线程Filter训练时间: {} 毫秒 ({:.3f} 秒)", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
    } else {
      printf("[DEBUG] Using single-threaded filter training. filter_train_is_mthread_ is: %d\n", config_.get().filter_train_is_mthread_);
      filter_train();
      auto filter_training_end_time = std::chrono::high_resolution_clock::now();
      auto filter_training_duration = std::chrono::duration_cast<std::chrono::milliseconds>(filter_training_end_time - filter_training_start_time);
      printf("单线程Filter训练时间: %ld 毫秒 (%.3f 秒)\n", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
      spdlog::info("单线程Filter训练时间: {} 毫秒 ({:.3f} 秒)", filter_training_duration.count(), filter_training_duration.count() / 1000.0);
    }

    // support difference devices for training and inference
    // 根据配置参数动态选择模型推理（inference）时使用的计算设备（CPU 或 GPU）
    if (config_.get().filter_infer_is_gpu_){
      printf("[DEBUG] Configuring inference device...\n");
      // TODO support multiple devices
      device_ = std::make_unique<torch::Device>(torch::kCUDA, static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
    } else {
      printf("[DEBUG] Using CPU for inference.\n");
      device_ = std::make_unique<torch::Device>(torch::kCPU);
    }
  }

  // ---------------------- 7. 导航器训练准备（收集叶子节点信息） ----------------------

  if (config_.get().navigator_is_learned_){
    leaf_nodes_.reserve(nleaf_);

    auto node_pos_to_id = make_reserved<ID_TYPE>(nleaf_);
    std::unordered_map<ID_TYPE, ID_TYPE> node_id_to_pos;
    node_id_to_pos.reserve(nleaf_ * 2);

    std::stack<std::reference_wrapper<dstree::Node>> node_cache;
    node_cache.push(std::ref(*root_));

    while (!node_cache.empty()){
      std::reference_wrapper<dstree::Node> node_to_visit = node_cache.top();
      node_cache.pop();

      if (node_to_visit.get().is_leaf()) {
        leaf_nodes_.push_back(node_to_visit);
        node_pos_to_id.push_back(node_to_visit.get().get_id());
        node_id_to_pos[node_to_visit.get().get_id()] = leaf_nodes_.size() - 1;
      } else {
        for (auto child_node : node_to_visit.get())
        {
          node_cache.push(child_node);
        }
      }
    }

    // 树结构遍历与叶子节点统计：代码遍历树结构，记录叶子节点，并统计每个训练样本的答案分布到叶子节点的频率（nn_residence_distributions）
    // 这本质上是计算每个查询对应的叶子节点被选中的概率分布。
    // 导航器训练：利用这些分布数据、查询特征和配置参数，训练一个导航器模型，可能用于预测查询在树结构中的路径或目标节点
    auto nn_residence_distributions = make_reserved<VALUE_TYPE>(config_.get().filter_train_nexample_ * nleaf_);
    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i){
      nn_residence_distributions.push_back(0);
    }

    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i){
      // use copy constructor to avoid destruct train_answers_
      Answers answers = Answers(train_answers_[query_i]);

      while (!answers.empty()){
        nn_residence_distributions[nleaf_ * query_i + node_id_to_pos[answers.pop_answer().node_id_]] += 1;
      }
    }

    for (ID_TYPE cell_i = 0; cell_i < config_.get().filter_train_nexample_ * nleaf_; ++cell_i){
      nn_residence_distributions[cell_i] /= config_.get().navigator_train_k_nearest_neighbor_;
    }

#ifdef DEBUG
    for (ID_TYPE query_i = 0; query_i < config_.get().filter_train_nexample_; ++query_i){
      spdlog::debug("navigator train query {:d} target = {:s}",
                    query_i, upcite::array2str(nn_residence_distributions.data() + nleaf_ * query_i, nleaf_));
    }
#endif

    navigator_ = std::make_unique<dstree::Navigator>(config_,
                                                     node_pos_to_id,
                                                     filter_train_query_tsr_,
                                                     nn_residence_distributions,
                                                     *device_);

    navigator_->train();
  }

  return SUCCESS;
}





RESPONSE dstree::Index::load(){
  ID_TYPE ifs_buf_size = sizeof(ID_TYPE) * config_.get().leaf_max_nseries_ * 2; // 2x expanded for safety
  ID_TYPE max_num_local_bytes = config_.get().filter_train_num_local_example_;
  if (config_.get().filter_train_num_global_example_ > max_num_local_bytes){
    max_num_local_bytes = config_.get().filter_train_num_global_example_;
  }
  max_num_local_bytes *= sizeof(VALUE_TYPE) * config_.get().series_length_;
  if (max_num_local_bytes > ifs_buf_size){
    ifs_buf_size = max_num_local_bytes;
  }

  void *ifs_buf = std::malloc(ifs_buf_size);
  nnode_ = 0;
  nleaf_ = 0;
  RESPONSE status = root_->load(ifs_buf, std::ref(*buffer_manager_), nnode_, nleaf_);
  std::free(ifs_buf);
  if (status == FAILURE){
    spdlog::info("failed to load index");
    return FAILURE;
  }
  // TODO in-memory only; supports on-disk
  if (!config_.get().on_disk_){
    printf("加载批处理数据到内存...\n");
    buffer_manager_->load_batch();
  }
  // 已删除leaf_min_heap_初始化 - 现在使用线程安全的局部优先队列

  // 这里应该是加载model
  printf("require_neurofilter_= %d\n", config_.get().require_neurofilter_);
  if (config_.get().require_neurofilter_){
    if (!config_.get().to_load_filters_){
      train();
    } else {
      if (config_.get().filter_retrain_){
        train(true);
      } else if (config_.get().filter_reallocate_multi_){
        // TODO
        filter_allocate(false, true);
      } else if (config_.get().filter_reallocate_single_){
        filter_allocate(false, true);
      } else {
        // initialize allocator for setting conformal intervals
        printf("---------进入load中的filter_allocate(false)分支----------\n");
        filter_allocate(false);
        // 在这里添加调试信息
        // printf("Loading filters...\n");
        // 这里需要一个函数来获取激活的过滤器数量
        printf("Number of active filters: %d\n", get_active_filter_count());
        // printf("Filter query filepath: %s\n", config_.get().filter_query_filepath_.c_str());
        printf("Filter load folder: %s\n", config_.get().index_load_folderpath_.c_str());

        // 添加：加载query_knn_nodes.bin文件
        printf("尝试加载查询KNN节点数据...\n");
        std::string knn_nodes_filepath = config_.get().index_load_folderpath_ + "query_knn_nodes.bin";
        
        if (fs::exists(knn_nodes_filepath)) {
            printf("找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
            std::ifstream knn_nodes_fin(knn_nodes_filepath, std::ios::binary);
            
            if (knn_nodes_fin.good()) {
                // 清空现有数据
                query_knn_nodes_.clear();
                
                // 读取查询数量
                ID_TYPE num_queries = 0;
                knn_nodes_fin.read(reinterpret_cast<char*>(&num_queries), sizeof(ID_TYPE));
                printf("文件中包含 %ld 个查询的KNN节点信息\n", static_cast<long>(num_queries));
                
                // 读取每个查询的节点信息
                for (ID_TYPE i = 0; i < num_queries; ++i) {
                    // 读取查询ID
                    ID_TYPE query_id = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&query_id), sizeof(ID_TYPE));
                    
                    // 读取该查询的节点映射大小
                    ID_TYPE num_nodes = 0;
                    knn_nodes_fin.read(reinterpret_cast<char*>(&num_nodes), sizeof(ID_TYPE));
                    
                    // 初始化该查询的节点映射
                    std::unordered_map<ID_TYPE, ID_TYPE> node_counts;
                    // 读取每个节点ID和计数
                    for (ID_TYPE j = 0; j < num_nodes; ++j){
                        ID_TYPE node_id = 0;
                        ID_TYPE count = 0;
                        knn_nodes_fin.read(reinterpret_cast<char*>(&node_id), sizeof(ID_TYPE));
                        knn_nodes_fin.read(reinterpret_cast<char*>(&count), sizeof(ID_TYPE));
                        
                        node_counts[node_id] = count;
                    }
                    // 存储到query_knn_nodes_中
                    query_knn_nodes_[query_id] = std::move(node_counts);
                }
                
                printf("成功加载查询KNN节点数据\n");
            } else {
                printf("无法打开查询KNN节点数据文件\n");
            }
        } else {
            printf("未找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
        }
        
        // 加载每个过滤器的batch_alphas.bin文件
        printf("尝试加载过滤器批处理alpha值...\n");
        
        // 遍历所有节点，查找激活的过滤器并加载其alpha值
        std::stack<std::reference_wrapper<dstree::Node>> node_stack;
        node_stack.push(std::ref(*root_));
        int loaded_filter_count = 0;
        
        while (!node_stack.empty()) {
            auto node = node_stack.top();
            node_stack.pop();
            
            if (node.get().is_leaf() && node.get().has_active_filter()) {
                ID_TYPE node_id = node.get().get_id();
                std::string alphas_filepath = config_.get().index_load_folderpath_ + 
                                             "filter_" + std::to_string(node_id) + "_alphas.bin";
                
                if (fs::exists(alphas_filepath)) {
                    // printf("加载过滤器 %ld 的批处理alpha值: %s\n", 
                    //        static_cast<long>(node_id), alphas_filepath.c_str());
                    
                    // 简化加载逻辑，不使用try-catch和额外的资源清理
                    if (node.get().load_filter_batch_alphas(alphas_filepath) == SUCCESS) {
                        loaded_filter_count++;
                        // printf("成功加载过滤器 %ld 的批处理alpha值\n", static_cast<long>(node_id));
                    } else {
                        printf("加载过滤器 %ld 的批处理alpha值失败\n", static_cast<long>(node_id));
                    }
                } else {
                    printf("未找到过滤器 %ld 的批处理alpha文件: %s\n", 
                           static_cast<long>(node_id), alphas_filepath.c_str());
                }
                
                // 加载批处理校准查询ID
                std::string calib_query_ids_filepath = config_.get().index_load_folderpath_ + 
                                                    "filter_" + std::to_string(node_id) + "_calib_query_ids.bin";
                                                    
                if (fs::exists(calib_query_ids_filepath)) {
                    // printf("加载过滤器 %ld 的批处理校准查询ID: %s\n", static_cast<long>(node_id), calib_query_ids_filepath.c_str());
                           
                    // 加载校准查询ID
                    RESPONSE calib_load_result = node.get().load_filter_batch_calib_query_ids(calib_query_ids_filepath);
                    if (calib_load_result == SUCCESS) {
                        // printf("成功加载过滤器 %ld 的批处理校准查询ID\n", static_cast<long>(node_id));
                    } else {
                        printf("加载过滤器 %ld 的批处理校准查询ID失败\n", static_cast<long>(node_id));
                    }
                } else {
                    printf("未找到过滤器 %ld 的批处理校准查询ID文件: %s\n", 
                           static_cast<long>(node_id), calib_query_ids_filepath.c_str());
                }
            }
            
            if (!node.get().is_leaf()) {
                for (auto child_node : node.get()) {
                    node_stack.push(child_node);
                }
            }
        }
        // printf("成功加载 %d 个过滤器的批处理alpha值\n", loaded_filter_count);
      }
      // support difference devices for training and inference
      if (config_.get().filter_infer_is_gpu_){
        // TODO support multiple devices
        device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                                  static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
      } else {
        device_ = std::make_unique<torch::Device>(torch::kCPU);
      }
    }
  }

  if (config_.get().navigator_is_learned_)
  {
    train();
  }
  return SUCCESS;
}



// 保存model
RESPONSE dstree::Index::dump() const{
  ID_TYPE ofs_buf_size = config_.get().filter_train_nexample_;
  if (config_.get().filter_train_num_global_example_ > ofs_buf_size){
    ofs_buf_size = config_.get().filter_train_num_global_example_;
  }
  if (config_.get().filter_train_num_local_example_ > ofs_buf_size){
    ofs_buf_size = config_.get().filter_train_num_local_example_;
  }
  if (ofs_buf_size < 1){
    ofs_buf_size = 128;
  }
  ofs_buf_size *= sizeof(ID_TYPE) * config_.get().series_length_;
  void *ofs_buf = std::malloc(ofs_buf_size);

  root_->dump(ofs_buf);
  // 保存 query_knn_nodes_ 数据
  if (config_.get().require_neurofilter_ && config_.get().filter_is_conformal_){
    std::string knn_nodes_filepath = config_.get().index_dump_folderpath_ + "query_knn_nodes.bin";
    std::ofstream knn_nodes_fout(knn_nodes_filepath, std::ios::binary);

    if (knn_nodes_fout.good()){
      // 保存 query_knn_nodes_ 的大小
      ID_TYPE num_queries = query_knn_nodes_.size();
      knn_nodes_fout.write(reinterpret_cast<const char *>(&num_queries), sizeof(ID_TYPE));

      // 保存每个查询的节点映射
      for (const auto &[query_id, node_counts] : query_knn_nodes_){
        // 保存查询ID
        knn_nodes_fout.write(reinterpret_cast<const char *>(&query_id), sizeof(ID_TYPE));

        // 保存该查询的节点映射大小
        ID_TYPE num_nodes = node_counts.size();
        knn_nodes_fout.write(reinterpret_cast<const char *>(&num_nodes), sizeof(ID_TYPE));

        // 保存每个节点ID和计数
        for (const auto &[node_id, count] : node_counts){
          knn_nodes_fout.write(reinterpret_cast<const char *>(&node_id), sizeof(ID_TYPE));
          knn_nodes_fout.write(reinterpret_cast<const char *>(&count), sizeof(ID_TYPE));
        }
      }

      // printf("保存查询KNN节点数据到 %s 成功\n", knn_nodes_filepath.c_str());
    } else {
      spdlog::error("无法打开文件保存查询KNN节点数据: {}", knn_nodes_filepath);
    }

    // 保存每个过滤器的 batch_alphas_ 数据
    // 遍历所有叶节点
    std::stack<std::reference_wrapper<dstree::Node>> node_stack;
    node_stack.push(std::ref(*root_));

    while (!node_stack.empty()){
      auto node = node_stack.top();
      node_stack.pop();

      if (node.get().is_leaf() && node.get().has_active_filter()){
        ID_TYPE node_id = node.get().get_id();
        std::string alphas_filepath = config_.get().index_dump_folderpath_ +
                                      "filter_" + std::to_string(node_id) + "_alphas.bin";

        // 获取过滤器的 conformal_predictor_ 并保存 batch_alphas_
        node.get().save_filter_batch_alphas(alphas_filepath);
        
        // 保存批处理校准查询ID
        std::string calib_query_ids_filepath = config_.get().index_dump_folderpath_ +
                                             "filter_" + std::to_string(node_id) + "_calib_query_ids.txt";
                                    
        // 获取批处理校准查询ID
        const auto& batch_calib_query_ids = node.get().get_filter().get().get_batch_calib_query_ids();
        if (!batch_calib_query_ids.empty()) {
          // 保存为文本格式
          std::ofstream calib_file(calib_query_ids_filepath);
          if (calib_file.good()) {
            // 写入批次数量作为注释
            calib_file << "# 批次数量: " << batch_calib_query_ids.size() << std::endl;
            
            // 写入每个批次的ID，每行一个批次
            for (const auto& batch : batch_calib_query_ids) {
              // 批次大小作为注释
              calib_file << "# 批次大小: " << batch.size() << std::endl;
              
              // 写入所有ID，用空格分隔
              for (size_t i = 0; i < batch.size(); ++i) {
                calib_file << batch[i];
                if (i < batch.size() - 1) {
                  calib_file << " ";
                }
              }
              calib_file << std::endl;
            }
            // printf("已保存过滤器 %ld 的校准批次ID到文本文件\n", static_cast<long>(node_id));
          } else {
            printf("无法打开文件保存过滤器 %ld 的校准批次ID\n", static_cast<long>(node_id));
          }
        }
      }

      if (!node.get().is_leaf()){
        for (auto child_node : node.get()){
          node_stack.push(child_node);
        }
      }
    }
  }

  std::free(ofs_buf);
  return SUCCESS;
}




  
void dstree::Index::store_bsf(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE bsf) {
    query_node_bsf_map_[query_id][node_id] = bsf;
}

VALUE_TYPE dstree::Index::get_bsf(ID_TYPE query_id, ID_TYPE node_id) {
    if (query_node_bsf_map_.count(query_id) && query_node_bsf_map_[query_id].count(node_id)) {
        return query_node_bsf_map_[query_id][node_id];
    }
    return constant::MAX_VALUE; // 默认值
}



// 修改后的search函数，返回查询记录向量
// 这个函数是用来计算query的knn节点的真实误差的，不是提前计算好的误差，用true_error=raw_predicted_distance - true_distance
std::vector<upcite::dstree::QueryPredictionRecord> dstree::Index::search_with_prediction_error(
    ID_TYPE query_id, 
    VALUE_TYPE *query_ptr, 
    VALUE_TYPE *sketch_ptr) {
  
  std::vector<QueryPredictionRecord> results;
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_) {
    route_ptr = sketch_ptr;
  }
  printf("Processing query_id: %ld\n", query_id);
  // 创建Answers对象，只需要最近邻 (K=1)
  auto answers = std::make_shared<dstree::Answers>(1, query_id);
  // auto answer = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);

  
  // 准备PyTorch张量用于神经网络输入
  torch::Tensor local_filter_query_tsr;
  if (config_.get().require_neurofilter_) {
    local_filter_query_tsr = torch::from_blob(
        query_ptr,
        {1, config_.get().series_length_},
        torch::TensorOptions().dtype(TORCH_VALUE_TYPE)
    ).to(*device_);
  }
  
  // 阶段一：找到初始叶子节点并搜索
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()) {
    resident_node = resident_node.get().route(route_ptr);
  }
  // auto stage1_start_time = std::chrono::high_resolution_clock::now();
  resident_node.get().search1(query_ptr, query_id, *answers);
  // auto stage1_end_time = std::chrono::high_resolution_clock::now();
  // auto stage1_duration = std::chrono::duration_cast<std::chrono::microseconds>(stage1_end_time - stage1_start_time);
  // spdlog::info("query id {} 初始访问节点 {} 耗时: {} 微秒", query_id, resident_node.get().get_id(), stage1_duration.count());
  // visited_series_counter_total += resident_node.get().get_size();
  
  // 使用局部优先队列进行线程安全搜索
  std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
  local_leaf_min_heap.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
  
  while (!local_leaf_min_heap.empty()) {
    auto heap_top = local_leaf_min_heap.top();
    std::reference_wrapper<dstree::Node> node_to_visit = std::get<0>(heap_top);
    VALUE_TYPE node2visit_lbdistance = std::get<1>(heap_top);
    local_leaf_min_heap.pop();

    
    if (node_to_visit.get().is_leaf()) {
      if (node_to_visit.get().get_id() != resident_node.get().get_id()) {
        if (answers->is_bsf(node2visit_lbdistance)) {
          // 获取最小bsf距离
          auto minbsf = answers->get_bsf();
          store_bsf(query_id, node_to_visit.get().get_id(), minbsf);
          // 将最小bsf距离添加到节点
          // node_to_visit.get().push_bsf_test_example(minbsf);
          // 搜索
          node_to_visit.get().search1(query_ptr, query_id, *answers);

        }
      }
    } else {
      for (auto child_node : node_to_visit.get()) {
        VALUE_TYPE child_lb = child_node.get().cal_lower_bound_EDsquare(route_ptr);
        local_leaf_min_heap.push(std::make_tuple(child_node, child_lb));
      }
    }
  }
  
  // 获取最近邻结果
  if (!answers->empty()) {
    auto topk = answers->get_current_topk();
    if (!topk.empty()) {
      QueryPredictionRecord record;
      record.query_id = query_id;

      // 这里真实距离应该开平方，但是预测距离不用(因为预测距离本身就是用开过平方的距离训练的模型)
      record.true_nn_distance = sqrt(topk[0].nn_dist_);  // 计算欧氏距离 (开平方)
      record.filter_id = topk[0].node_id_;
      record.series_id = topk[0].global_offset_; //index中query id的最近邻对应的序列
      // auto& node = topk[0].node_id_.get();
      // auto& filter = node.get_filter().get();
      // record.bsf_distance = filter.get_bsf_distance_test(query_id);
      record.bsf_distance = sqrt(get_bsf(query_id, topk[0].node_id_));
      // 获取该节点的过滤器预测
      VALUE_TYPE predicted_distance = 0.0;
      
      // 找到对应的节点引用
      std::stack<std::reference_wrapper<dstree::Node>> node_stack;
      node_stack.push(std::ref(*root_));
      bool found_node = false;
      
      while (!node_stack.empty() && !found_node) {
        auto current = node_stack.top();
        node_stack.pop();
        
        if (current.get().get_id() == record.filter_id) {
          // 找到对应节点，获取预测距离
          if (current.get().has_active_filter()) {
            //！！！！预测距离不用开根号
            predicted_distance = sqrt(current.get().filter_infer_raw(local_filter_query_tsr));
            found_node = true;
          
          } else {
            // 节点没有激活的过滤器
            predicted_distance = -1.0;  // 表示没有预测
            found_node = true;
            
          }
        }
        
        // 如果是内部节点且尚未找到目标节点，继续搜索子节点
        if (!current.get().is_leaf() && !found_node) {
          for (auto child : current.get()) {
            node_stack.push(child);
          }
        }
      }
      
      record.predicted_distance = predicted_distance;
      record.prediction_error = predicted_distance - record.true_nn_distance;
      
      // 打印结果
      printf("Query %ld: true_distance=%.4f, predicted=%.4f, error=%.4f, filter_id=%ld, series_id=%ld, bsf_distance=%.4f\n",
             query_id, record.true_nn_distance, record.predicted_distance, 
             record.prediction_error, record.filter_id, record.series_id, record.bsf_distance);
      results.push_back(record);

    }
  }
  return results;
}



// 收集所有查询的最近邻预测和真实距离信息
RESPONSE dstree::Index::collect_all_query_prediction_data(const std::string& output_filepath) {
  // 1. 前置检查：确保查询文件存在
  if (!fs::exists(config_.get().query_filepath_)) {
    spdlog::error("查询文件不存在: {:s}", config_.get().query_filepath_);
    return FAILURE;
  }
  
  printf("开始收集所有查询的最近邻预测数据...\n");
  printf("查询文件路径: %s\n", config_.get().query_filepath_.c_str());
  
  // 2. 打开查询文件
  std::ifstream query_fin(config_.get().query_filepath_, std::ios::in | std::ios::binary);
  if (!query_fin.good()) {
    spdlog::error("无法打开查询文件: {:s}", config_.get().query_filepath_);
    return FAILURE;
  }
  
  // 3. 读取查询数据
  auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().query_nseries_;
  auto query_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_nbytes));
  query_fin.read(reinterpret_cast<char *>(query_buffer), query_nbytes);
  
  if (query_fin.fail()) {
    spdlog::error("读取查询数据失败: {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);
    std::free(query_buffer);
    return FAILURE;
  }
  
  // 4. 读取草图数据（如果提供）
  VALUE_TYPE *query_sketch_buffer = nullptr;
  if (config_.get().is_sketch_provided_) {
    if (!fs::exists(config_.get().query_sketch_filepath_)) {
      spdlog::error("草图文件不存在: {:s}", config_.get().query_sketch_filepath_);
      std::free(query_buffer);
      return FAILURE;
    }
    
    std::ifstream query_sketch_fin(config_.get().query_sketch_filepath_, std::ios::in | std::ios::binary);
    if (!query_sketch_fin.good()) {
      spdlog::error("无法打开草图文件: {:s}", config_.get().query_sketch_filepath_);
      std::free(query_buffer);
      return FAILURE;
    }
    
    auto query_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * config_.get().query_nseries_;
    query_sketch_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_sketch_nbytes));
    query_sketch_fin.read(reinterpret_cast<char *>(query_sketch_buffer), query_sketch_nbytes);
    
    // 错误处理
    if (query_sketch_fin.fail()){
      spdlog::error("cannot read {:d} bytes from {:s}", query_sketch_nbytes, config_.get().query_sketch_filepath_);
      std::free(query_sketch_buffer);
      return FAILURE;
    }
  }
  // 确保输出目录存在
  {
    namespace fs = boost::filesystem;
    fs::path out_path(output_filepath);
    fs::path parent_dir = out_path.parent_path();
    if (!parent_dir.empty() && !fs::exists(parent_dir)) {
        fs::create_directories(parent_dir);           // 确保目录存在
    }
  }
  // 5. 准备输出文件
  std::ofstream results_file(output_filepath);
  if (!results_file.is_open()) {
    spdlog::error("无法创建输出文件: {:s}", output_filepath);
    std::free(query_buffer);
    if (query_sketch_buffer) std::free(query_sketch_buffer);
    return FAILURE;
  }
  
  // 写入CSV头
  results_file << "query_id,filter_id,true_nn_distance,predicted_distance,prediction_error,series_id,bsf_distance\n";
  
  // 6. 为每个查询执行搜索并收集数据
  printf("开始处理查询...\n");
  for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
    // 提取当前查询数据
    VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
    VALUE_TYPE *sketch_ptr = nullptr;
    if (config_.get().is_sketch_provided_) {
      sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
    }
    
    // 执行搜索并收集预测数据
    auto start_time = std::chrono::high_resolution_clock::now();
    auto prediction_records = search_with_prediction_error(query_id, series_ptr, sketch_ptr);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    // 打印每个查询的执行时间
    
    printf("查询 %ld 执行时间: %ld 微秒\n", query_id, duration.count());
    spdlog::info("查询 %ld 执行时间: %ld 微秒", query_id, duration.count());
    
    // 将结果写入文件
    for (const auto& record : prediction_records) {
      results_file << record.query_id << ","
                  << record.filter_id << ","
                  << record.true_nn_distance << ","
                  << record.predicted_distance << ","
                  << record.prediction_error << ","
                  << record.series_id << ","
                  << record.bsf_distance << "\n";
    }
    
    // 每处理100个查询打印一次进度
    if (query_id % 100 == 0 || query_id == config_.get().query_nseries_ - 1) {
      printf("已处理 %ld/%ld 个查询\n", query_id + 1, config_.get().query_nseries_);
    }
  }
  
  // 关闭文件并清理资源
  results_file.close();
  std::free(query_buffer);
  if (query_sketch_buffer) std::free(query_sketch_buffer);
  
  printf("所有查询处理完成，结果已保存到: %s\n", output_filepath.c_str());
  return SUCCESS;
}




// query 搜索阶段
RESPONSE dstree::Index::search(bool is_profile){
  // 添加调试信息显示is_profile参数值
  printf("=== Search函数调试信息 ===\n");
  printf("is_profile参数值: %s\n", is_profile ? "true (profile模式)" : "false (正常搜索模式)");
  printf("========================\n");
  
  if (is_profile) {
    printf("注意: 当前为profile模式，将跳过recall计算和CSV生成！\n");
    printf("如需生成CSV文件，请确保is_profile=false\n");
  }
  
  // ==================== 1. 前置检查模块 ====================
  // 检查查询文件是否存在
  if (!fs::exists(config_.get().query_filepath_)){
    spdlog::error("query filepath {:s} does not exist", config_.get().query_filepath_);
    return FAILURE;
  }
  printf(" ---------- Index::search ---------- \n");
  // 尝试打开查询文件
  std::ifstream query_fin(config_.get().query_filepath_, std::ios::in | std::ios::binary);
  if (!query_fin.good()){
    spdlog::error("query filepath {:s} cannot open", config_.get().query_filepath_);
    return FAILURE;
  }
  // ==================== 2. 读取查询数据模块 ====================
  // 计算需要读取的字节数并分配内存
  auto query_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().series_length_ * config_.get().query_nseries_;
  auto query_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_nbytes));
  // 从文件读取原始查询数据
  query_fin.read(reinterpret_cast<char *>(query_buffer), query_nbytes);

  if (query_fin.fail()){
    spdlog::error("cannot read {:d} bytes from {:s}", query_nbytes, config_.get().query_filepath_);
    return FAILURE;
  }

  // ==================== 3. 读取草图数据模块（没用到）====================
  VALUE_TYPE *query_sketch_buffer = nullptr;
  if (config_.get().is_sketch_provided_){
    // 检查草图文件是否存在  PAA=sketch
    if (!fs::exists(config_.get().query_sketch_filepath_)){
      spdlog::error("query sketch filepath {:s} does not exist", config_.get().query_sketch_filepath_);
      return FAILURE;
    }

    // 打开并读取草图数据
    std::ifstream query_sketch_fin(config_.get().query_sketch_filepath_, std::ios::in | std::ios::binary);
    if (!query_fin.good()){
      spdlog::error("query sketch filepath {:s} cannot open", config_.get().query_sketch_filepath_);
      return FAILURE;
    }

    auto query_sketch_nbytes = static_cast<ID_TYPE>(sizeof(VALUE_TYPE)) * config_.get().sketch_length_ * config_.get().query_nseries_;
    query_sketch_buffer = static_cast<VALUE_TYPE *>(std::malloc(query_sketch_nbytes));
    query_sketch_fin.read(reinterpret_cast<char *>(query_sketch_buffer), query_sketch_nbytes);

    // 错误处理
    if (query_sketch_fin.fail()){
      spdlog::error("cannot read {:d} bytes from {:s}", query_sketch_nbytes, config_.get().query_sketch_filepath_);
      return FAILURE;
    }
  }



  // ==================== 4. 算法核心调整模块 ====================
  printf("config_.get().filter_conformal_adjust_confidence_by_recall_ = %d\n",config_.get().filter_conformal_adjust_confidence_by_recall_);
  if (config_.get().require_neurofilter_ && config_.get().filter_is_conformal_ && config_.get().filter_conformal_adjust_confidence_by_recall_){
    auto adjust_confidence_start_time = std::chrono::high_resolution_clock::now();
    
    // allocator_.get()->set_batch_confidence_from_recall(this->get_query_knn_nodes());
    // 在index.cc或其他地方
    // 将unique_ptr转换为shared_ptr
    std::shared_ptr<dstree::Node> shared_root(root_.get(), [](dstree::Node*){});  //  total error  使用自定义删除器避免重复删除
    // allocator_.get()->simulate_full_search_for_recall(shared_root); //max error?

    allocator_.get()->simulate_full_search_for_recall_alpha_based(shared_root); // total error
    // allocator_.get()->document_cp_dist(this->get_query_knn_nodes());
    
    auto adjust_confidence_end_time = std::chrono::high_resolution_clock::now();
    auto adjust_confidence_duration = std::chrono::duration_cast<std::chrono::microseconds>(adjust_confidence_end_time - adjust_confidence_start_time);
    printf("算法核心调整模块执行时间: %.3f 秒\n", adjust_confidence_duration.count() / 1000000.0);
    spdlog::info("算法核心调整模块执行时间: {:.3f} 秒", adjust_confidence_duration.count() / 1000000.0);
  }


  printf("\n---------执行搜索和计算Recall模块------------\n");
  // ==================== 5. 执行搜索和计算Recall模块 ====================
  if (!is_profile){ // 当不是profile模式时，执行recall计算
    // ==================== 新增：Ground Truth结果管理 ====================
    if (!fs::exists(config_.get().ground_truth_path_)) {
      fs::create_directories(config_.get().ground_truth_path_);
      printf("创建Ground Truth目录: %s\n", config_.get().ground_truth_path_.c_str());
      spdlog::info("创建Ground Truth目录: {}", config_.get().ground_truth_path_);
    }
    std::string gt_filepath = config_.get().ground_truth_path_ + "/ground_truth_results.bin";
    bool gt_loaded = false;
    
    // 检查是否存在已保存的ground truth结果
    if (fs::exists(gt_filepath)) {
      printf("发现已保存的Ground Truth结果，尝试加载: %s\n", gt_filepath.c_str());
      spdlog::info("发现已保存的Ground Truth结果，尝试加载: {}", gt_filepath);
      
      if (load_ground_truth_results(gt_filepath, query_buffer) == SUCCESS) {
        
        printf("Ground Truth结果加载成功，跳过暴力搜索阶段\n");
        spdlog::info("Ground Truth结果加载成功，跳过暴力搜索阶段");
        gt_loaded = true;
      } else {
        printf("Ground Truth结果加载失败，将重新进行暴力搜索\n");
        spdlog::warn("Ground Truth结果加载失败，将重新进行暴力搜索");
        gt_loaded = false;
        ground_truth_answers_.clear(); // 只有加载失败时才清空
      }
    } else {
      printf("未找到已保存的Ground Truth结果，将进行暴力搜索\n");
      spdlog::info("未找到已保存的Ground Truth结果，将进行暴力搜索");
      gt_loaded = false;
      ground_truth_answers_.clear(); // 只有没有文件时才清空
    }
    
    actual_answers_.clear();
    // 初始化统计变量
    std::vector<ID_TYPE> gt_visited_series(config_.get().query_nseries_, 0);    // 暴力搜索每个查询访问的序列数
    std::vector<ID_TYPE> filter_visited_series_total(config_.get().query_nseries_, 0); // 过滤器搜索每个查询访问的序列数
    std::vector<ID_TYPE> filter_pruned_series_calib(config_.get().query_nseries_, 0); // 过滤器剪枝的序列数
    std::vector<ID_TYPE> filter_pruned_series_lb(config_.get().query_nseries_, 0); // 过滤器剪枝的序列数
    
    // 总计统计变量
    ID_TYPE total_gt_series = 0;
    ID_TYPE total_pruned_series_calib = 0;
    ID_TYPE total_pruned_series_lb = 0;
    ID_TYPE total_visited_series_total = 0;

    if (!gt_loaded) {
      printf("=== Phase 1: 执行多线程暴力搜索获取ground truth ===\n");
      spdlog::info("=== Phase 1: 执行多线程暴力搜索获取ground truth ===");
      
      // 记录暴力搜索所有query的总时间
      auto gt_search_start_time = std::chrono::high_resolution_clock::now();

      // 确定线程数量 - 固定设置为20个线程
      int num_threads = 40;  // 固定使用20个线程
      printf("使用 %d 个线程进行暴力搜索\n", num_threads);
      spdlog::info("使用 {} 个线程进行暴力搜索", num_threads);

      // 预先分配 ground_truth_answers_ 的空间
      ground_truth_answers_.resize(config_.get().query_nseries_);
      
      // 创建线程来处理不同的query范围
      std::vector<std::thread> threads;
      std::vector<std::vector<ID_TYPE>> thread_gt_visited_series(num_threads);
      
      // 为每个线程创建本地的ground_truth_answers存储
      std::vector<std::vector<std::shared_ptr<dstree::Answers>>> thread_gt_answers(num_threads);
      
      // 计算每个线程处理的query数量
      ID_TYPE queries_per_thread = config_.get().query_nseries_ / num_threads;
      ID_TYPE remaining_queries = config_.get().query_nseries_ % num_threads;
      
      for (int t = 0; t < num_threads; ++t) {
        ID_TYPE start_query = t * queries_per_thread;
        ID_TYPE end_query = (t + 1) * queries_per_thread;
        if (t == num_threads - 1) {
          end_query += remaining_queries; // 最后一个线程处理剩余的query
        }
        
        // 每个线程的统计数组和结果数组
        thread_gt_visited_series[t].resize(end_query - start_query, 0);
        thread_gt_answers[t].resize(end_query - start_query);
        
        threads.emplace_back([this, t, start_query, end_query, &thread_gt_visited_series, 
                             &thread_gt_answers, query_buffer, query_sketch_buffer]() {
          
          for (ID_TYPE query_id = start_query; query_id < end_query; ++query_id) {
            VALUE_TYPE *sketch_ptr = nullptr;
            if (config_.get().is_sketch_provided_) {
              sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
            }
            
            // 创建答案对象并执行暴力搜索
            VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
            auto gt_answers = new dstree::Answers(config_.get().n_nearest_neighbor_, query_id);
            ID_TYPE current_visited_nodes = 0, current_visited_series = 0;
            
            // 阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
            profile(query_id, series_ptr, sketch_ptr, gt_answers, current_visited_nodes, current_visited_series);
            
            // 存储到线程本地结果
            thread_gt_answers[t][query_id - start_query] = std::shared_ptr<dstree::Answers>(gt_answers);
            
            // 记录此次查询的统计信息到线程本地数组
            thread_gt_visited_series[t][query_id - start_query] = current_visited_series;
          }
        });
      }
      
      // 等待所有线程完成
      for (auto& thread : threads) {
        thread.join();
      }
      
      // 一次性汇总ground_truth_answers_和统计信息
      for (int t = 0; t < num_threads; ++t) {
        ID_TYPE start_query = t * queries_per_thread;
        // 添加维度检查
        assert(thread_gt_visited_series[t].size() == thread_gt_answers[t].size());
        for (size_t i = 0; i < thread_gt_visited_series[t].size(); ++i) {
          gt_visited_series[start_query + i] = thread_gt_visited_series[t][i];
          ground_truth_answers_[start_query + i] = std::move(thread_gt_answers[t][i]);
          total_gt_series += thread_gt_visited_series[t][i];
        }
      }
      
      auto gt_search_end_time = std::chrono::high_resolution_clock::now();
      auto gt_search_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(gt_search_end_time - gt_search_start_time);
      printf("多线程暴力搜索所有 %d 个query总耗时: %ld 微秒\n", config_.get().query_nseries_, gt_search_total_duration.count());
      spdlog::info("多线程暴力搜索所有 {} 个query总耗时: {} 微秒", config_.get().query_nseries_, gt_search_total_duration.count());
      
      if (save_ground_truth_results(gt_filepath, query_buffer) == SUCCESS) {
        printf("Ground Truth结果已保存到: %s\n", gt_filepath.c_str());
        spdlog::info("Ground Truth结果已保存到: {}", gt_filepath);
      } else {
        printf("警告：Ground Truth结果保存失败\n");
        spdlog::warn("Ground Truth结果保存失败");
      }
    }

    
    printf("===== Phase 2: 执行多线程带过滤器的实际搜索 =====\n");
    spdlog::info("=== Phase 2: 执行多线程带过滤器的实际搜索 ===");
    
    // 记录过滤器搜索所有query的总时间
    auto filter_search_start_time = std::chrono::high_resolution_clock::now();
    
    // 确定线程数量
    int num_threads = 40;  // 固定使用20个线程
    printf("使用 %d 个线程进行过滤器搜索\n", num_threads);
    spdlog::info("使用 {} 个线程进行过滤器搜索", num_threads);
    
    // 预先分配 actual_answers_ 的空间
    actual_answers_.resize(config_.get().query_nseries_);
    
    // 创建线程来处理不同的query范围
    std::vector<std::thread> filter_threads;
    std::vector<std::vector<ID_TYPE>> thread_filter_visited_series_total(num_threads);
    std::vector<std::vector<ID_TYPE>> thread_filter_pruned_series_calib(num_threads);
    std::vector<std::vector<ID_TYPE>> thread_filter_pruned_series_lb(num_threads);
    
    // 为每个线程创建本地的actual_answers存储
    std::vector<std::vector<std::shared_ptr<dstree::Answers>>> thread_actual_answers(num_threads);
    
    // 计算每个线程处理的query数量
    ID_TYPE queries_per_thread = config_.get().query_nseries_ / num_threads;
    ID_TYPE remaining_queries = config_.get().query_nseries_ % num_threads;
    
    for (int t = 0; t < num_threads; ++t) {
      ID_TYPE start_query = t * queries_per_thread;
      ID_TYPE end_query = (t + 1) * queries_per_thread;
      if (t == num_threads - 1) {
        end_query += remaining_queries; // 最后一个线程处理剩余的query
      }
      
      // 每个线程的统计数组和结果数组
      thread_filter_visited_series_total[t].resize(end_query - start_query, 0);
      thread_filter_pruned_series_calib[t].resize(end_query - start_query, 0);
      thread_filter_pruned_series_lb[t].resize(end_query - start_query, 0);
      thread_actual_answers[t].resize(end_query - start_query);
      
      filter_threads.emplace_back([this, t, start_query, end_query, 
                                  &thread_filter_visited_series_total, 
                                  &thread_filter_pruned_series_calib,
                                  &thread_filter_pruned_series_lb,
                                  &thread_actual_answers,
                                  query_buffer, query_sketch_buffer]() {
        
        for (ID_TYPE query_id = start_query; query_id < end_query; ++query_id) {
          // 遍历所有的query, 计算每个query的knn结果，存放在actual_answers     
          VALUE_TYPE *sketch_ptr = nullptr;
          if (config_.get().is_sketch_provided_) {
            sketch_ptr = query_sketch_buffer + config_.get().sketch_length_ * query_id;
          }

          // 创建答案对象并执行实际搜索
          VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
          auto actual_answers = new dstree::Answers(config_.get().n_nearest_neighbor_, query_id);
          ID_TYPE current_visited_nodes = 0, current_visited_series_total = 0;
          ID_TYPE current_nfpruned_nodes_calib = 0, current_nfpruned_series_calib = 0;
          ID_TYPE current_nfpruned_nodes_lb = 0, current_nfpruned_series_lb = 0;

          // !!!!!!!! 执行带有过滤器的搜索 !!!!!!
          // 创建本地的filter_query_tensor用于多线程安全搜索
          torch::Tensor local_filter_query_tensor;
          if (config_.get().require_neurofilter_) {
            local_filter_query_tensor = torch::from_blob(series_ptr, {1, config_.get().series_length_},
                                                        torch::kFloat32).to(*device_);
          }
          // 注意：当require_neurofilter_=false时，local_filter_query_tensor保持未初始化状态
          // 但search函数内部会检查require_neurofilter_，不会使用它
          
          search(query_id, series_ptr, sketch_ptr, actual_answers, current_visited_nodes, current_visited_series_total, 
                 current_nfpruned_nodes_calib, current_nfpruned_series_calib,
                 current_nfpruned_nodes_lb, current_nfpruned_series_lb, local_filter_query_tensor);
          
          // 存储到线程本地结果
          thread_actual_answers[t][query_id - start_query] = std::shared_ptr<dstree::Answers>(actual_answers);
          
          // 记录此次查询的统计信息到线程本地数组
          thread_filter_visited_series_total[t][query_id - start_query] = current_visited_series_total;
          thread_filter_pruned_series_calib[t][query_id - start_query] = current_nfpruned_series_calib;
          thread_filter_pruned_series_lb[t][query_id - start_query] = current_nfpruned_series_lb;

          // 减少密集日志输出 - 仅每100个query打印一次进度
          if (query_id % 100 == 0) {
            printf("线程 %d 完成查询 %ld\n", t, query_id);
          }
        }
      });
    }
    
    // 等待所有线程完成
    for (auto& thread : filter_threads) {
      thread.join();
    }
    
    // 一次性汇总actual_answers_和统计信息
    for (int t = 0; t < num_threads; ++t) {
      ID_TYPE start_query = t * queries_per_thread;
      // 添加维度检查
      assert(thread_filter_visited_series_total[t].size() == thread_actual_answers[t].size());
      assert(thread_filter_pruned_series_calib[t].size() == thread_actual_answers[t].size());
      assert(thread_filter_pruned_series_lb[t].size() == thread_actual_answers[t].size());
      
      for (size_t i = 0; i < thread_filter_visited_series_total[t].size(); ++i) {
        filter_visited_series_total[start_query + i] = thread_filter_visited_series_total[t][i];
        filter_pruned_series_calib[start_query + i] = thread_filter_pruned_series_calib[t][i];
        filter_pruned_series_lb[start_query + i] = thread_filter_pruned_series_lb[t][i];
        actual_answers_[start_query + i] = std::move(thread_actual_answers[t][i]);
        
        total_visited_series_total += thread_filter_visited_series_total[t][i];
        total_pruned_series_calib += thread_filter_pruned_series_calib[t][i];
        total_pruned_series_lb += thread_filter_pruned_series_lb[t][i];
      }
    }
    
    auto filter_search_end_time = std::chrono::high_resolution_clock::now();
    auto filter_search_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(filter_search_end_time - filter_search_start_time);
    printf("多线程过滤器搜索所有 %d 个query总耗时: %.6f 秒\n", config_.get().query_nseries_, filter_search_total_duration.count() / 1000000.0);
    spdlog::info("多线程过滤器搜索所有 {} 个query总耗时: {} 微秒", config_.get().query_nseries_, filter_search_total_duration.count());

    // 仅对前10个查询输出详细结果，避免日志过多
    printf("===== 输出前10个查询的实际结果 =====\n");
    for (ID_TYPE query_id = 0; query_id < std::min(static_cast<ID_TYPE>(10), config_.get().query_nseries_); ++query_id) {
      if (actual_answers_[query_id]) {
        auto actual_topk = actual_answers_[query_id]->get_current_topk();
        for (size_t i = 0; i < std::min(static_cast<size_t>(3), actual_topk.size()); ++i) {
          printf("query_id = %ld, K=%zu: distance=%.4f, node ID=%ld, series id=%ld\n", 
                 query_id, i+1, sqrt(actual_topk[i].nn_dist_), actual_topk[i].node_id_, actual_topk[i].global_offset_);
        }
      }
    }

    // 将查询结果写入文件，便于对比
    // std::string results_path = config_.get().save_path_;
    // // 确保结果目录存在
    // if (!results_path.empty()) {
    //   namespace fs = boost::filesystem;
    //   if (!fs::exists(results_path)) {
    //     printf("创建结果保存目录: %s\n", results_path.c_str());
    //     fs::create_directories(results_path);
    //   }
    // }

    // // 更改为CSV格式以便于Excel打开
    // std::string comparison_file = results_path + "/query_results_comparison.csv";
    
    // // 添加调试信息，检查数据容器状态
    // printf("=== CSV文件生成调试信息 ===\n");
    // printf("ground_truth_answers_.size(): %zu\n", ground_truth_answers_.size());
    // printf("actual_answers_.size(): %zu\n", actual_answers_.size());
    // printf("config_.get().query_nseries_: %ld\n", (long)config_.get().query_nseries_);
    
    // if (ground_truth_answers_.empty()) {
    //   printf("警告: ground_truth_answers_ 为空！\n");
    //   spdlog::warn("ground_truth_answers_ 为空！");
    // }
    
    // if (actual_answers_.empty()) {
    //   printf("警告: actual_answers_ 为空！\n");
    //   spdlog::warn("actual_answers_ 为空！");
    // }
    
    // std::ofstream result_file(comparison_file);
    // if (result_file.is_open()) {
    //   // CSV表头
    //   result_file << "query_id,K,暴力距离,过滤器距离,暴力节点ID,过滤器节点ID,暴力offset,过滤器offset,是否匹配\n";
      
    //   int total_rows_written = 0;
      
    //   // 遍历所有查询
    //   for (ID_TYPE query_id = 0; query_id < config_.get().query_nseries_; ++query_id) {
    //     // 检查索引是否有效
    //     if (query_id >= ground_truth_answers_.size()) {
    //       printf("错误: query_id %ld 超出 ground_truth_answers_ 范围 (%zu)\n", 
    //              query_id, ground_truth_answers_.size());
    //       break;
    //     }
        
    //     if (query_id >= actual_answers_.size()) {
    //       printf("错误: query_id %ld 超出 actual_answers_ 范围 (%zu)\n", 
    //              query_id, actual_answers_.size());
    //       break;
    //     }
        
    //     auto& gt_answers = ground_truth_answers_[query_id];
    //     auto& actual_answers = actual_answers_[query_id];
        
    //     // 检查指针是否有效
    //     if (!gt_answers) {
    //       printf("错误: ground_truth_answers_[%ld] 为空指针\n", query_id);
    //       continue;
    //     }
        
    //     if (!actual_answers) {
    //       printf("错误: actual_answers_[%ld] 为空指针\n", query_id);
    //       continue;
    //     }
        
    //     auto gt_topk = gt_answers->get_current_topk();
    //     auto actual_topk = actual_answers->get_current_topk();
        
    //     // 添加更详细的调试信息
    //     if (query_id < 5) { // 只打印前5个查询的详细信息
    //       printf("查询 %ld: GT结果数 = %zu, 实际结果数 = %zu\n", 
    //              query_id, gt_topk.size(), actual_topk.size());
    //     }
        
    //     ID_TYPE k_max = std::min(gt_topk.size(), actual_topk.size());
        
    //     for (ID_TYPE k = 0; k < k_max; ++k) {
    //       bool is_match = (gt_topk[k].global_offset_ == actual_topk[k].global_offset_);
    //       result_file << query_id << "," << (k+1) << ","
    //                  << gt_topk[k].nn_dist_ << "," << actual_topk[k].nn_dist_ << ","
    //                  << gt_topk[k].node_id_ << "," << actual_topk[k].node_id_ << ","
    //                  << gt_topk[k].global_offset_ << "," << actual_topk[k].global_offset_ << ","
    //                  << (is_match ? "是" : "否") << "\n";
    //       total_rows_written++;
    //     }
    //   }
      
    //   result_file.close();
    //   printf("CSV文件写入完成，共写入 %d 行数据\n", total_rows_written);
    //   printf("查询结果对比已保存到: %s\n", comparison_file.c_str());
    //   spdlog::info("CSV文件写入完成，共写入 {} 行数据", total_rows_written);
    //   spdlog::info("查询结果对比已保存到: {}", comparison_file);

    // } else {

    //   printf("错误：无法创建结果对比文件 %s\n", comparison_file.c_str());
    //   spdlog::error("无法创建结果对比文件 {}", comparison_file);
    // }
    

    
    auto recall_start = std::chrono::high_resolution_clock::now();
    // 计算recall
    calculate_batch_recall();

    auto recall_end = std::chrono::high_resolution_clock::now();
    auto recall_duration = std::chrono::duration_cast<std::chrono::microseconds>(recall_end - recall_start);
    printf("计算recall总耗时: %ld 微秒\n", recall_duration.count());
    spdlog::info("计算recall总耗时: {} 微秒", recall_duration.count());

    // 计算总的剪枝率（calib + lb）
    double total_pruning_ratio = total_visited_series_total > 0 ? 
                                    static_cast<double>(total_pruned_series_calib + total_pruned_series_lb) / total_visited_series_total : 0.0;
 
    
    // 计算不同类型的剪枝率
    double calib_prune_ratio = total_visited_series_total > 0 ? 
                              static_cast<double>(total_pruned_series_calib) / total_visited_series_total : 0.0;
    
    double lb_prune_ratio = total_visited_series_total > 0 ? 
                           static_cast<double>(total_pruned_series_lb) / total_visited_series_total : 0.0;
    

    printf("\n===== node visited stats =====\n");
    printf("total_ground_truth_series: %ld\n", total_gt_series);
    printf("total_visited_series_total: %ld\n", total_visited_series_total);

    printf("total_pruning_ratio: %.4f\n", total_pruning_ratio);
    printf("calib_prune_ratio: %.4f\n", calib_prune_ratio);
    printf("lb_prune_ratio: %.4f\n", lb_prune_ratio);
    
    spdlog::info("总访问序列数: {}",  total_visited_series_total);
    spdlog::info("总剪枝序列数: {}", total_pruned_series_calib + total_pruned_series_lb);

    spdlog::info("总体剪枝比例: {:.4f}", total_pruning_ratio);
    spdlog::info("校准剪枝比例: {:.4f}", calib_prune_ratio);
    spdlog::info("下界剪枝比例: {:.4f}", lb_prune_ratio);
    
  } 
  // 释放内存
  std::free(query_buffer);
  if (query_sketch_buffer != nullptr){
    std::free(query_sketch_buffer);
  }
  printf("[DEBUG] ----- recall end ----- \n");
  return SUCCESS;
}





// 执行暴力搜索 - 带访问统计版本 - 多线程安全版本
RESPONSE dstree::Index::profile(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                                dstree::Answers *results, 
                                ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter_total){
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  // 初始化统计变量
  visited_node_counter = 0;
  visited_series_counter_total = 0;

  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    //实际走的是这个分支
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  // 注意：profile函数执行暴力搜索，不需要过滤器，所以不创建torch张量

  //从根节点开始，逐步向下遍历树结构，直到找到包含目标查询序列的叶子节点。
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()){
    //这步快速找到最可能包含相似序列的初始叶子节点
    resident_node = resident_node.get().route(route_ptr);
  }
  //阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
  resident_node.get().search1(query_ptr, query_id, *answers);
  visited_series_counter_total += resident_node.get().get_size();

  //阶段二：精确搜索：对于同一个query，使用优先队列(leaf_min_heap_)按下界距离从小到大检查其他节点
  if (config_.get().is_exact_search_){
    // 使用本地优先队列避免线程冲突
    std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
    local_leaf_min_heap.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;

    while (!local_leaf_min_heap.empty()){
      std::tie(node_to_visit, node2visit_lbdistance) = local_leaf_min_heap.top();
      local_leaf_min_heap.pop();

      if (node_to_visit.get().is_leaf()){
          //is_bsf 用于判断lb_distance是否小于minbsf
          if (node_to_visit.get().get_id() != resident_node.get().get_id()){
            // 如果lb<minbsf，则进行搜索； 如果下列if语句不满足，则剪枝
              if (answers->is_bsf(node2visit_lbdistance)) {
                node_to_visit.get().search1(query_ptr, query_id, *answers);
              }
          // 这个统计的是不使用任何剪枝时需要访问的序列总数
          visited_node_counter += 1;
          visited_series_counter_total += node_to_visit.get().get_size();
          }

      } else {
        for (auto child_node : node_to_visit.get()){
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
          local_leaf_min_heap.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }
    }
  }

  return SUCCESS;
}








// 执行带过滤器的实际搜：带访问统计版本 - 多线程安全版本
RESPONSE dstree::Index::search(ID_TYPE query_id, VALUE_TYPE *query_ptr, VALUE_TYPE *sketch_ptr,
                              dstree::Answers *results,
                              ID_TYPE &visited_node_counter, ID_TYPE &visited_series_counter_total,
                              ID_TYPE &nfpruned_node_counter_calib, ID_TYPE &nfpruned_series_counter_calib,
                              ID_TYPE &nfpruned_node_counter_lb, ID_TYPE &nfpruned_series_counter_lb,
                              torch::Tensor &filter_query_tensor){
  // 添加三个计数器统计各种情况
  int counter_a = 0; // lb无法剪枝: filter剪枝成功的次数
  int counter_b = 0; // lb无法剪枝: filter不能剪枝的次数
  int counter_c = 0; // lb无法剪枝: 没有激活filter, 开始暴搜的次数
  int counter_d = 0; // lb可以剪枝: 利用lb进行剪枝的次数
  int counter_search = 0; // 实际搜索节点的次数
  VALUE_TYPE *route_ptr = query_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  // spdlog::info("query_id: {}", query_id);
  fflush(stdout);
  
  // 初始化计数器变量
  visited_node_counter = 0;
  visited_series_counter_total = 0;
  nfpruned_node_counter_calib = 0;
  nfpruned_series_counter_calib = 0;
  nfpruned_node_counter_lb = 0;
  nfpruned_series_counter_lb = 0;
  
  // 修改：使用传入的results对象或创建新的
  std::shared_ptr<dstree::Answers> answers;
  if (results != nullptr){
    answers = std::shared_ptr<dstree::Answers>(results, [](dstree::Answers *) {}); // 非拥有指针
  } else {
    answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
  }

  //从根节点开始，逐步向下遍历树结构，直到找到包含目标查询序列的叶子节点。
  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);
  while (!resident_node.get().is_leaf()){
    //这步快速找到最可能包含相似序列的初始叶子节点
    resident_node = resident_node.get().route(route_ptr);
  }
  //阶段一： 近似搜索，查找当前节点resident_node下距离query_ptr最近距离，更新bsf
  VALUE_TYPE nndistance = resident_node.get().search1(query_ptr, query_id, *answers);
  counter_search += 1;
  
  visited_series_counter_total += resident_node.get().get_size();

  //阶段二：精确搜索：对于同一个query，使用优先队列(leaf_min_heap_)按下界距离从小到大检查其他节点
  if (config_.get().is_exact_search_){
    // 使用本地优先队列，避免线程冲突
    std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist> local_leaf_min_heap;
    local_leaf_min_heap.push(std::make_tuple(std::ref(*root_), root_->cal_lower_bound_EDsquare(route_ptr)));
    std::reference_wrapper<dstree::Node> node_to_visit = std::ref(*(dstree::Node *)nullptr);
    VALUE_TYPE node2visit_lbdistance;
    
    while (!local_leaf_min_heap.empty()){
      std::tie(node_to_visit, node2visit_lbdistance) = local_leaf_min_heap.top();
      local_leaf_min_heap.pop();

      if (node_to_visit.get().is_leaf()){
          if (node_to_visit.get().get_id() != resident_node.get().get_id()){
           
         
            if (answers->is_bsf(node2visit_lbdistance)) {
              // 1. lb不能剪枝
              // lb<minbsf时，返回true， 考虑用predict_distance剪枝
              if (node_to_visit.get().has_active_filter() && config_.get().require_neurofilter_){
                
                VALUE_TYPE predicted_nn_distance = node_to_visit.get().filter_infer_calibrated(filter_query_tensor); // 这个距离是用预测距离-校准误差得到的校准后的距离

                // 为了加速计算，这里没有开平方，pred_distance只要>=0, 同时minbsf>0, 此时开平方和不开去比大小，结果是一样的
                if (predicted_nn_distance > answers->get_bsf()){
                    // 1.1 filter 剪枝  当校准距离>minbsf时，则用校准距离进行剪枝
                    nfpruned_node_counter_calib += 1;
                    nfpruned_series_counter_calib += node_to_visit.get().get_size();
                } else {
                    // 1.2 暴力搜索  当校准距离 <= minbsf时，(lb和校准距离都剪枝不了）
                    VALUE_TYPE nn_dist = node_to_visit.get().search1(query_ptr, query_id, *answers);
                    counter_search += 1;
                }

              } else {
                  // 1.3 暴力搜索  当filter不激活时，则用暴力搜索 (lb和校准距离都剪枝不了）
                  VALUE_TYPE nn_dist = node_to_visit.get().search1(query_ptr, query_id, *answers);
                  counter_search += 1;
                  // auto end_time = std::chrono::high_resolution_clock::now();
                  // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                  // spdlog::info("query id {} 访问节点 {} 耗时: {} 微秒", query_id, node_to_visit.get().get_id(), duration.count());
              }
              
            } else {
                // 2. lower bound 剪枝   lb>minbsf，则用lb进行剪枝
                // counter_d++;
                nfpruned_node_counter_lb += 1;
                nfpruned_series_counter_lb += node_to_visit.get().get_size();
            }

            // lb和predict_distance都无法剪枝，visited_series_counter_total是不使用任何剪枝时需要访问的序列总数
            visited_node_counter += 1;
            visited_series_counter_total += node_to_visit.get().get_size();
          }
          
      } else {
        // printf("\n==========query_id: %ld 访问到 节点 %d 非叶子节点==========\n", query_id, node_to_visit.get().get_id());

        // 对于非叶子节点，将子节点加入优先队列
        for (auto child_node : node_to_visit.get()){
          VALUE_TYPE child_lower_bound_EDsquare = child_node.get().cal_lower_bound_EDsquare(route_ptr);
          local_leaf_min_heap.push(std::make_tuple(child_node, child_lower_bound_EDsquare));
        }
      }

    }

  }

  return SUCCESS;
}




// 计算Recall的实现
void dstree::Index::calculate_recall(){
  double total_recall = 0.0;

  for (ID_TYPE query_id = 0; query_id < ground_truth_answers_.size(); ++query_id){
    printf("\n==========query_id: %ld 开始计算recall==========\n", query_id);
    auto &gt_result = ground_truth_answers_[query_id];
    auto &actual_result = actual_answers_[query_id];

    // 提取ground truth的global_offset集合
    std::set<ID_TYPE> gt_offsets;
    auto gt_copy = *gt_result; // 创建副本以避免修改原始对象

    while (!gt_copy.empty()){
      auto answer = gt_copy.pop_answer();
      gt_offsets.insert(answer.global_offset_); // 使用global_offset作为唯一标识符
    }

    // 检查实际结果中有多少匹配的offset
    int hits = 0;
    auto actual_copy = *actual_result; // 创建副本

    while (!actual_copy.empty()){
      auto answer = actual_copy.pop_answer();
      if (gt_offsets.find(answer.global_offset_) != gt_offsets.end()){
        hits++;
      }
    }

    // 计算当前查询的recall
    double recall = static_cast<double>(hits) / config_.get().n_nearest_neighbor_;
    total_recall += recall;

    // printf("Query %ld Recall: %.4f (hits: %d/%ld)\n",
    //        query_id, recall, hits, config_.get().n_nearest_neighbor_);
  }

  // 计算平均recall
  double avg_recall = total_recall / ground_truth_answers_.size(); // 使用正确的变量名
  printf("\n total avgRecall: %.4f\n", avg_recall);
}





// 计算Recall的实现(分批次)
void dstree::Index::calculate_batch_recall(){
  // 确保结果保存路径存在
  std::string save_path = config_.get().save_path_;
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      printf("创建结果保存目录: %s\n", save_path.c_str());
      fs::create_directories(save_path);
    }
  }

  // 修改：每个batch包含100个query，根据总query数量自动计算batch数量
  int queries_per_batch, num_batches;
  
  if (ground_truth_answers_.size() <= 100) {
    // 如果总query数量小于100，就用一个batch包含所有query
    queries_per_batch = 10;
    num_batches = static_cast<int>(std::floor(static_cast<double>(ground_truth_answers_.size()) / queries_per_batch));
  } else {
    // 如果总query数量>100，按100一个batch计算，向下取整
    queries_per_batch = 100;
    num_batches = static_cast<int>(std::floor(static_cast<double>(ground_truth_answers_.size()) / queries_per_batch));
  }

  printf("\n===== 按批次计算Recall（每批次%d个query，共%d个批次）=====\n", queries_per_batch, num_batches);
  spdlog::info("===== 按批次计算Recall（每批次{}个query，共{}个批次）=====", queries_per_batch, num_batches);
  std::vector<double> batch_recalls(num_batches, 0.0);
  std::vector<int> batch_query_counts(num_batches, 0);
  double total_recall = 0.0;

  // 获取目标recall和coverage配置
  double target_recall = config_.get().filter_conformal_recall_;
  double target_coverage = config_.get().filter_conformal_coverage_;
  
  printf("target Recall: %.2f, target coverage: %.2f\n", target_recall, target_coverage);
  spdlog::info("目标Recall: {:.2f}, 目标覆盖率: {:.2f}", target_recall, target_coverage);
  
  // // 打开文件保存每个query的recall
  // std::string recall_file = config_.get().save_path_ + "/query_recalls.txt";
  // std::ofstream recall_ofs(recall_file);
  // if (!recall_ofs.is_open()) {
  //   printf("警告: 无法创建recall文件: %s\n", recall_file.c_str());
  // } else {
  //   recall_ofs << "query_id,hits,recall\n"; // CSV格式表头
  // }
  
  for (ID_TYPE query_id = 0; query_id < ground_truth_answers_.size(); ++query_id) {
    int batch_id = query_id / queries_per_batch; // 确定当前查询属于哪个批次
    if (batch_id >= num_batches)
      batch_id = num_batches - 1; // 安全检查

    auto &gt_result = ground_truth_answers_[query_id];
    auto &actual_result = actual_answers_[query_id];

    // 提取ground truth的结果，保持原始顺序
    auto gt_copy = *gt_result; // 创建副本以避免修改原始对象
    auto actual_copy = *actual_result; // 创建副本

    std::vector<upcite::Answer> gt_answers;
    std::set<ID_TYPE> gt_offsets;
    
    // 保存ground truth答案的顺序
    while (!gt_copy.empty()) {
      auto answer = gt_copy.pop_answer();
      gt_answers.push_back(answer);
      gt_offsets.insert(answer.global_offset_); // 同时构建offset集合
    }
    
    // 反转以恢复从小到大的顺序
    std::reverse(gt_answers.begin(), gt_answers.end());
    
    // 获取actual答案并保持顺序
    std::vector<upcite::Answer> actual_answers;
    while (!actual_copy.empty()) {
      auto answer = actual_copy.pop_answer();
      actual_answers.push_back(answer);
    }
    
    // 反转以恢复从小到大的顺序
    std::reverse(actual_answers.begin(), actual_answers.end());
    
    // 检查matches和异常情况
    int hits = 0;
    int hits_with_same_dist = 0; // 新增：匹配且距离相等的计数
    size_t min_size = std::min(gt_answers.size(), actual_answers.size());
    
    for (size_t i = 0; i < min_size; i++) {
      if (gt_answers[i].global_offset_ == actual_answers[i].global_offset_) {
        hits++;
        // 新增：检查距离是否相等
        if (std::abs(gt_answers[i].nn_dist_ - actual_answers[i].nn_dist_) < 1e-6) {
          hits_with_same_dist++;
        } else {
          // 输出距离不同的警告
          printf("\n注意: 查询 %ld 的第 %zu 个结果匹配但距离不同:\n", query_id, i+1);
          printf("  Ground Truth: offset=%ld, 距离=%.6f\n", 
                 gt_answers[i].global_offset_, sqrt(gt_answers[i].nn_dist_));
          printf("  Actual Result: offset=%ld, 距离=%.6f\n", 
                 actual_answers[i].global_offset_, sqrt(actual_answers[i].nn_dist_));
        }
      } else {
        // 不匹配时检查距离
        if (actual_answers[i].nn_dist_ < gt_answers[i].nn_dist_) {
          // 这是异常情况，actual距离小于gt距离
          printf("\n警告: 查询 %ld 的第 %zu 近邻出现异常情况:\n", query_id, i+1);
          printf("  Ground Truth: offset=%ld, 距离=%.4f, 节点=%ld\n", 
                 gt_answers[i].global_offset_, sqrt(gt_answers[i].nn_dist_), gt_answers[i].node_id_);
          printf("  Actual Result: offset=%ld, 距离=%.4f, 节点=%ld\n", 
                 actual_answers[i].global_offset_, sqrt(actual_answers[i].nn_dist_), actual_answers[i].node_id_);
          printf("  Actual距离小于Ground Truth距离，这是不正常的!\n");
        }
      }
    }

    // 计算当前查询的recall
    double recall = static_cast<double>(hits) / config_.get().n_nearest_neighbor_;
    total_recall += recall;
    
    // 新增：输出匹配且距离相等的数量
    // printf("查询 %ld: 匹配数=%d, 其中距离相等=%d\n", query_id, hits, hits_with_same_dist);

    // 累加到对应批次
    batch_recalls[batch_id] += recall;
    batch_query_counts[batch_id]++;

    // // 写入recall文件
    // if (recall_ofs.is_open()) {
    //   recall_ofs << query_id << "," << hits << "," << recall << "\n";
    // }
  }

  // // 关闭recall文件
  // if (recall_ofs.is_open()) {
  //   recall_ofs.close();
  //   printf("已保存%zu个query的recall到文件: %s\n", ground_truth_answers_.size(), recall_file.c_str());
  //   spdlog::info("已保存{}个query的recall到文件: {}", ground_truth_answers_.size(), recall_file);
  // }

  // 打印每个批次的平均recall
  printf("\n===== avgRecall for top-10 batches ... =====\n");
  int batches_above_target = 0;
  int valid_batches = 0;
  
  for (int batch_id = 0; batch_id < num_batches; ++batch_id) {
    if (batch_query_counts[batch_id] > 0) {
      valid_batches++;
      double batch_avg_recall = batch_recalls[batch_id] / batch_query_counts[batch_id];
      
       // 将两个数四舍五入到4位小数后比较
      double rounded_batch_recall = std::round(batch_avg_recall * 100000) / 100000.0;
      double rounded_target = std::round(target_recall * 100000) / 100000.0;
      bool above_target = (rounded_batch_recall >= rounded_target);
      
      // const double epsilon = 1e-4;  // 小的误差容忍度
      // bool above_target = (batch_avg_recall >= target_recall - epsilon);
      if (above_target) {
        batches_above_target++;
      }
      // 只打印前10个批次的信息
      if (batch_id < 10) {
        printf("batch %d (query %d-%d): avgRecall = %.4f %s\n",
               batch_id + 1,
               batch_id * queries_per_batch,
               std::min(static_cast<ID_TYPE>((batch_id + 1) * queries_per_batch - 1), static_cast<ID_TYPE>(ground_truth_answers_.size() - 1)),
               batch_avg_recall,
               above_target ? "[Success]" : "[Failed]");
      }

      spdlog::info("Batch {} (Queries {} - {}): Average Recall = {:.4f} {}",
                   batch_id + 1,
                   batch_id * queries_per_batch,
                   std::min(static_cast<ID_TYPE>((batch_id + 1) * queries_per_batch - 1), static_cast<ID_TYPE>(ground_truth_answers_.size() - 1)),
                   batch_avg_recall,
                   above_target ? "[Passed]" : "[Failed]");   
    }
  }

  // 计算总体平均recall
  double avg_recall = total_recall / ground_truth_answers_.size();
  
  // 计算达标批次的比例
  double achieved_coverage = static_cast<double>(batches_above_target) / valid_batches;
  
  printf("target recall: %.4f, target coverage: %.4f\n", target_recall, target_coverage);
  printf("\nOverall Average Recall: %.4f\n", avg_recall);
  printf("achieved coverage: %.4f \n", achieved_coverage);
  spdlog::info("target recall: {:.4f}, target coverage: {:.4f}", target_recall, target_coverage);
  spdlog::info("Overall Average Recall: {:.4f}", avg_recall);
  spdlog::info("Achieved coverage: {:.4f}", achieved_coverage);
  
         
  // 判断是否同时满足recall和coverage的要求
  bool recall_passed = (avg_recall >= target_recall);
  bool coverage_passed = (achieved_coverage >= target_coverage);
  
  if (recall_passed && coverage_passed) {
    printf("Conclusion: [PASSED] Recall requirement: yes, Coverage requirement: yes\n");
    spdlog::info("Conclusion: [PASSED] Recall requirement: yes, Coverage requirement: yes");
  } else if (recall_passed && !coverage_passed) {
    printf("Conclusion: [FAILED] Recall requirement: yes, Coverage requirement: No\n");
    spdlog::error("Conclusion: [FAILED] Recall requirement: yes, Coverage requirement: No");
  } else if (!recall_passed && coverage_passed) {
    printf("Conclusion: [FAILED] Recall requirement: No, Coverage requirement: yes\n");
    spdlog::error("Conclusion: [FAILED] Recall requirement: No, Coverage requirement: yes");
  } else {
    printf("Conclusion: [FAILED] Recall requirement: No, Coverage requirement: No\n");
    spdlog::error("Conclusion: [FAILED] Recall requirement: No, Coverage requirement: No");
  }
}






// 保存ground truth结果到文件
RESPONSE dstree::Index::save_ground_truth_results(const std::string& filepath, VALUE_TYPE* query_buffer) const {
  printf("保存ground truth结果到文件: %s\n", filepath.c_str());
  
  std::ofstream ofs(filepath, std::ios::binary);
  if (!ofs.is_open()) {
    printf("错误: 无法创建ground truth结果文件: %s\n", filepath.c_str());
    return FAILURE;
  }

  // 写入文件格式版本
  uint32_t version = 1;
  ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));

  // 写入结果数量
  uint32_t num_queries = ground_truth_answers_.size();
  ofs.write(reinterpret_cast<const char*>(&num_queries), sizeof(num_queries));

  // 写入每个查询的结果
  for (size_t query_id = 0; query_id < ground_truth_answers_.size(); ++query_id) {
    auto& result = ground_truth_answers_[query_id];
    
    // 创建结果副本以便遍历
    auto result_copy = *result;
    std::vector<upcite::Answer> answers;
    
    // 提取所有答案
    while (!result_copy.empty()) {
      answers.push_back(result_copy.pop_answer());
    }
    
    // 反转以保持正确顺序
    std::reverse(answers.begin(), answers.end());
    // 写入这个查询的答案数量
    uint32_t num_answers = answers.size();
    ofs.write(reinterpret_cast<const char*>(&num_answers), sizeof(num_answers));
    // printf("DEBUG SAVE: query_id=%zu, num_answers=%u\n", query_id, num_answers);
    // 写入每个答案
    for (const auto& answer : answers) {
      ofs.write(reinterpret_cast<const char*>(&query_id), sizeof(query_id));                         // 新增：保存query_id
      
      // 新增：保存query时间序列的前几个值作为验证标识
      VALUE_TYPE *series_ptr = query_buffer + config_.get().series_length_ * query_id;
      uint32_t validation_length = std::min(static_cast<uint32_t>(4), static_cast<uint32_t>(config_.get().series_length_)); // 保存前4个值作为验证
      ofs.write(reinterpret_cast<const char*>(&validation_length), sizeof(validation_length));
      ofs.write(reinterpret_cast<const char*>(series_ptr), sizeof(VALUE_TYPE) * validation_length);
      ofs.write(reinterpret_cast<const char*>(&answer.global_offset_), sizeof(answer.global_offset_));
      ofs.write(reinterpret_cast<const char*>(&answer.nn_dist_), sizeof(answer.nn_dist_));
      ofs.write(reinterpret_cast<const char*>(&answer.node_id_), sizeof(answer.node_id_));
      // printf("DEBUG SAVE: query id %zu, global_offset=%u, nn_dist=%.6f, node_id=%u\n", 
      //        query_id, answer.global_offset_, answer.nn_dist_, answer.node_id_);
    }
  }
  ofs.close();
  printf("成功保存%zu个查询的ground truth结果\n", ground_truth_answers_.size());
  return SUCCESS;
}



















RESPONSE dstree::Index::search_navigated(ID_TYPE query_id, VALUE_TYPE *series_ptr, VALUE_TYPE *sketch_ptr){
  VALUE_TYPE *route_ptr = series_ptr;
  if (config_.get().is_sketch_provided_){
    route_ptr = sketch_ptr;
  }

  auto answers = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);

  if (config_.get().require_neurofilter_ || config_.get().navigator_is_learned_){
    filter_query_tsr_ = torch::from_blob(series_ptr, {1, config_.get().series_length_},torch::TensorOptions().dtype(TORCH_VALUE_TYPE)).to(*device_);
  }

  std::reference_wrapper<dstree::Node> resident_node = std::ref(*root_);

  while (!resident_node.get().is_leaf()){
    resident_node = resident_node.get().route(route_ptr);
  }

  ID_TYPE visited_node_counter = 0, visited_series_counter = 0;
  ID_TYPE nfpruned_node_counter = 0, nfpruned_series_counter = 0;

  resident_node.get().search1(series_ptr, query_id, *answers);

  if (config_.get().is_exact_search_){
    auto node_prob = navigator_->infer(filter_query_tsr_);
    auto node_distances = make_reserved<VALUE_TYPE>(nleaf_);

    if (config_.get().navigator_is_combined_){
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (leaf_nodes_[leaf_i].get().get_id() == resident_node.get().get_id()){
          node_distances.push_back(constant::MAX_VALUE);
        } else {
          node_distances.push_back(leaf_nodes_[leaf_i].get().cal_lower_bound_EDsquare(route_ptr));
        }
      }

      VALUE_TYPE min_prob = constant::MAX_VALUE, max_prob = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (node_prob[leaf_i] < min_prob)
        {
          min_prob = node_prob[leaf_i];
        }
        else if (node_prob[leaf_i] > max_prob)
        {
          max_prob = node_prob[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        node_prob[leaf_i] = (node_prob[leaf_i] - min_prob) / (max_prob - min_prob);
      }

      VALUE_TYPE min_lb_dist = constant::MAX_VALUE, max_lb_dist = constant::MIN_VALUE;
      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        if (node_distances[leaf_i] < min_lb_dist)
        {
          min_lb_dist = node_distances[leaf_i];
        }
        else if (node_distances[leaf_i] > max_lb_dist && node_distances[leaf_i] < constant::MAX_VALUE / 2)
        {
          max_lb_dist = node_distances[leaf_i];
        }
      }

      for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i){
        node_prob[leaf_i] = config_.get().navigator_combined_lambda_ * node_prob[leaf_i] + (1 - config_.get().navigator_combined_lambda_) * (1 - (node_distances[leaf_i] - min_lb_dist) / (max_lb_dist - min_lb_dist));
      }
    }

    auto node_pos_probs = make_reserved<std::tuple<ID_TYPE, VALUE_TYPE>>(nleaf_);

    for (ID_TYPE leaf_i = 0; leaf_i < nleaf_; ++leaf_i)
    {
      if (leaf_nodes_[leaf_i].get().get_id() != resident_node.get().get_id())
      {
        node_pos_probs.push_back(std::tuple<ID_TYPE, VALUE_TYPE>(leaf_i, node_prob[leaf_i]));
      }
    }

#ifdef DEBUG
    // #ifndef DEBUGGED
    spdlog::debug("query {:d} node_distances = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_distances.data(), nleaf_));

    spdlog::debug("query {:d} node_prob = {:s}",
                  answers.get()->query_id_, upcite::array2str(node_prob.data(), nleaf_));
// #endif
#endif

    std::sort(node_pos_probs.begin(), node_pos_probs.end(), dstree::compDecrProb);

    for (ID_TYPE prob_i = 0; prob_i < node_pos_probs.size(); ++prob_i)
    {
      ID_TYPE leaf_i = std::get<0>(node_pos_probs[prob_i]);
      auto node_to_visit = leaf_nodes_[leaf_i];

      // #ifdef DEBUG
      ////#ifndef DEBUGGED
      //      spdlog::debug("query {:d} leaf_i {:d} ({:d}) dist {:.3f} prob {:.3f} ({:.3f}) bsf {:.3f}",
      //                    answers.get()->query_id_,
      //                    leaf_i, navigator_->get_id_from_pos(leaf_i),
      //                    node_distances[leaf_i],
      //                    std::get<1>(node_pos_probs[prob_i]), node_prob[leaf_i],
      //                    answers->get_bsf());
      ////#endif
      // #endif

      if (visited_node_counter < config_.get().search_max_nnode_ &&
          visited_series_counter < config_.get().search_max_nseries_)
      {
        if (config_.get().examine_ground_truth_ || answers->is_bsf(node_distances[leaf_i]))
        {
          node_to_visit.get().search1(series_ptr, query_id, *answers);
        }
      }
    }
  }

  spdlog::info("query {:d} visited {:d} nodes {:d} series",
               query_id, visited_node_counter, visited_series_counter);

  ID_TYPE nnn_to_return = config_.get().n_nearest_neighbor_;

  while (!answers->empty())
  {
    auto answer = answers->pop_answer();

    if (answer.node_id_ > 0)
    {
      printf("query %d nn %d = %.3f, node %d, db_global_offset_ %d\n",
             query_id, nnn_to_return, answer.nn_dist_, answer.node_id_, answer.global_offset_);
    }
    else
    {
      printf("query %d nn %d = %.3f, db_global_offset_ %d\n",
             query_id, nnn_to_return, answer.nn_dist_, answer.global_offset_);
    }
    nnn_to_return -= 1;
  }

  if (nnn_to_return > 0)
  {
    return FAILURE;
  }

  return SUCCESS;
}

// 获取激活的过滤器数量
int dstree::Index::get_active_filter_count() const
{
  int count = 0;
  if (root_)
  {
    count_active_filters(*root_, count);
  }
  return count;
}

// 递归计算激活的过滤器数量
void dstree::Index::count_active_filters(const Node &node, int &count) const
{
  if (node.has_active_filter())
  {
    count++;
  }

  if (!node.is_leaf())
  {
    // 不能直接用迭代器，因为 begin() 和 end() 不是 const 方法
    // 为了解决这个问题，我们可以使用 node 的 children_ 成员变量
    // 但由于这是私有成员，我们需要改用其他方法判断

    // 如果节点不是叶子节点，使用递归方式获取所有子节点
    // 创建一个临时的非const Node以便访问其子节点
    Node &non_const_node = const_cast<Node &>(node);
    for (auto &child : non_const_node)
    {
      count_active_filters(child.get(), count);
    }
  }
}



// 新增函数，从预测error文件加载alpha值
RESPONSE dstree::Index::load_filter_errors_from_file(const std::string& error_file_path) {
  printf("\n===== 从文件加载filter预测error值 =====\n");
  printf("加载文件路径: %s\n", error_file_path.c_str());

  // 检查文件是否存在
  if (!fs::exists(error_file_path)) {
    printf("错误: 预测error文件不存在: %s\n", error_file_path.c_str());
    return FAILURE;
  }

  // 打开文件
  std::ifstream fin(error_file_path);
  if (!fin.is_open() || !fin.good()) {
    printf("错误: 无法打开预测error文件: %s\n", error_file_path.c_str());
    return FAILURE;
  }

  // 读取文件头
  std::string header;
  std::getline(fin, header);
  printf("文件头: %s\n", header.c_str());

  // 创建filter_id到error值的映射
  std::unordered_map<ID_TYPE, VALUE_TYPE> filter_errors;
  
  // 读取每一行数据
  std::string line;
  int line_count = 0;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    std::string filter_id_str, recall_str, coverage_str, error_str;
    
    // 按逗号分割行
    std::getline(iss, filter_id_str, ',');
    std::getline(iss, recall_str, ',');
    std::getline(iss, coverage_str, ',');
    std::getline(iss, error_str, ',');
    
    // 转换为数值
    ID_TYPE filter_id = std::stoi(filter_id_str);
    VALUE_TYPE error = std::stof(error_str);
    
    // 存储到映射中
    filter_errors[filter_id] = error;
    line_count++;
  }
  
  printf("从文件中读取了 %d 个filter的error值\n", line_count);
  
  // 使用栈进行树的遍历，为每个active_filter设置alpha值
  std::stack<std::reference_wrapper<dstree::Node>> node_stack;
  node_stack.push(std::ref(*root_));
  int updated_filter_count = 0;
  
  while (!node_stack.empty()) {
    auto node = node_stack.top();
    node_stack.pop();
    
    if (node.get().is_leaf() && node.get().has_active_filter()) {
      ID_TYPE node_id = node.get().get_id();
      
      // 检查该filter_id是否在映射中
      if (filter_errors.find(node_id) != filter_errors.end()) {
        VALUE_TYPE error_value = filter_errors[node_id];
        
        // 设置该filter的alpha_值
        auto& filter = node.get().get_filter().get();
        filter.set_alpha_directly(error_value);
        
        // printf("设置filter %ld的alpha_值为 %.3f\n", (long)node_id, error_value);
        updated_filter_count++;
      } else {
        printf("警告: 未找到filter %ld 的error值\n", (long)node_id);
      }
    }
    
    // 将非叶子节点的子节点加入栈中继续遍历
    if (!node.get().is_leaf()) {
      for (auto child_node : node.get()) {
        node_stack.push(child_node);
      }
    }
  }
  
  printf("成功更新 %d 个filter的alpha_值\n", updated_filter_count);
  return SUCCESS;
}




// 增强版加载函数，特别针对批处理alpha值和查询KNN节点数据
RESPONSE dstree::Index::load_enhanced(){
  printf("===== 开始使用增强版加载函数 =====\n");
  
  // ==== 第一部分：基本缓冲区设置 ====
  // 计算所需的缓冲区大小
  ID_TYPE ifs_buf_size = sizeof(ID_TYPE) * config_.get().leaf_max_nseries_ * 2; // 基础大小
  ID_TYPE max_num_local_bytes = config_.get().filter_train_num_local_example_;
  
  // 确保缓冲区足够大以处理全局或本地示例
  if (config_.get().filter_train_num_global_example_ > max_num_local_bytes){
    max_num_local_bytes = config_.get().filter_train_num_global_example_;
  }
  max_num_local_bytes *= sizeof(VALUE_TYPE) * config_.get().series_length_;
  
  // 使用更大的缓冲区
  if (max_num_local_bytes > ifs_buf_size){
    ifs_buf_size = max_num_local_bytes;
  }

  printf("分配读取缓冲区: %ld 字节\n", (long)ifs_buf_size);
  void *ifs_buf = std::malloc(ifs_buf_size);
  if (ifs_buf == nullptr) {
    spdlog::error("内存分配失败，无法继续加载");
    return FAILURE;
  }

  // ==== 第二部分：加载树结构 ====
  printf("开始加载树结构...\n");
  nnode_ = 0;
  nleaf_ = 0;
  RESPONSE status = root_->load(ifs_buf, std::ref(*buffer_manager_), nnode_, nleaf_);
  
  if (status == FAILURE){
    spdlog::error("加载索引失败");
    std::free(ifs_buf);
    return FAILURE;
  }
  
  printf("成功加载树结构: %ld 个节点, %ld 个叶子节点\n", (long)nnode_, (long)nleaf_);
  
  // ==== 第三部分：加载批处理数据 ====
  // 仅针对内存模式
  if (!config_.get().on_disk_){
    printf("加载批处理数据到内存...\n");
    buffer_manager_->load_batch();
  }
  // 移除叶节点最小堆初始化 - 现在使用线程安全的本地优先队列
  // leaf_min_heap_ = std::priority_queue<NODE_DISTNCE, std::vector<NODE_DISTNCE>, CompareDecrNodeDist>(
  //     CompareDecrNodeDist(), make_reserved<dstree::NODE_DISTNCE>(nleaf_));
  
  // ==== 第四部分：神经过滤器处理 ====
  printf("神经过滤器设置: require_neurofilter=%d, to_load_filters=%d\n", 
         config_.get().require_neurofilter_, config_.get().to_load_filters_);
  
  if (config_.get().require_neurofilter_){
    if (!config_.get().to_load_filters_){
      printf("不加载已有过滤器，执行新训练...\n");
      train();
    } else {
      if (config_.get().filter_retrain_){
        printf("使用已保存的初始化执行重新训练...\n");
        train(true);
      } else if (config_.get().filter_reallocate_multi_ || config_.get().filter_reallocate_single_){
        printf("执行过滤器重新分配...\n");
        filter_allocate(false, true);
      } else {
        // ==== 第五部分：加载过滤器及相关数据 ====
        printf("===== 加载过滤器相关数据 =====\n");
        printf("初始化过滤器分配器...\n");
        filter_allocate(false);
        
        // 获取激活的过滤器数量
        int active_filter_count = get_active_filter_count();
        printf("激活的过滤器数量: %d\n", active_filter_count);
        printf("加载目录: %s\n", config_.get().index_load_folderpath_.c_str());
        
        // // 5.1 加载查询KNN节点数据
        // printf("\n===== 加载查询KNN节点数据 =====\n");
        // std::string knn_nodes_filepath = config_.get().index_load_folderpath_ + "query_knn_nodes.bin";
        bool knn_nodes_loaded = false;
        
        // if (fs::exists(knn_nodes_filepath)) {
        //     printf("找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
        //     std::ifstream knn_nodes_fin(knn_nodes_filepath, std::ios::binary);
            
        //     if (knn_nodes_fin.good()) {
        //         // 清空现有数据
        //         query_knn_nodes_.clear();
                
        //         // 读取查询数量
        //         ID_TYPE num_queries = 0;
        //         knn_nodes_fin.read(reinterpret_cast<char*>(&num_queries), sizeof(ID_TYPE));
        //         printf("文件中包含 %ld 个查询的KNN节点信息\n", static_cast<long>(num_queries));
                
        //         // 读取每个查询的节点信息
        //         for (ID_TYPE i = 0; i < num_queries; ++i) {
        //             // 读取查询ID
        //             ID_TYPE query_id = 0;
        //             knn_nodes_fin.read(reinterpret_cast<char*>(&query_id), sizeof(ID_TYPE));
                    
        //             // 读取该查询的节点映射大小
        //             ID_TYPE num_nodes = 0;
        //             knn_nodes_fin.read(reinterpret_cast<char*>(&num_nodes), sizeof(ID_TYPE));
                    
        //             // 初始化该查询的节点映射
        //             std::unordered_map<ID_TYPE, ID_TYPE> node_counts;
        //             // 读取每个节点ID和计数
        //             for (ID_TYPE j = 0; j < num_nodes; ++j){
        //                 ID_TYPE node_id = 0;
        //                 ID_TYPE count = 0;
        //                 knn_nodes_fin.read(reinterpret_cast<char*>(&node_id), sizeof(ID_TYPE));
        //                 knn_nodes_fin.read(reinterpret_cast<char*>(&count), sizeof(ID_TYPE));
                        
        //                 node_counts[node_id] = count;
        //             }
                    
        //             // 存储到query_knn_nodes_中
        //             query_knn_nodes_[query_id] = std::move(node_counts);
        //         }
                
        //         printf("成功加载 %zu 个查询的KNN节点数据\n", query_knn_nodes_.size());
        //         knn_nodes_loaded = true;
        //     } else {
        //         printf("无法打开查询KNN节点数据文件\n");
        //     }
        // } else {
        //     printf("未找到查询KNN节点数据文件: %s\n", knn_nodes_filepath.c_str());
        // }
        
        if (!knn_nodes_loaded) {
            printf("注意: 未能加载查询KNN节点数据，某些功能可能不可用\n");
        }
        
        // 5.2 加载过滤器的批处理alpha值
        printf("\n===== 加载过滤器批处理alpha值 =====\n");
        
        // 使用栈进行树的遍历
        // std::stack<std::reference_wrapper<dstree::Node>> node_stack;
        // node_stack.push(std::ref(*root_));
        int loaded_filter_count = 0;
        int attempted_load_count = 0;
        bool any_calib_query_ids_loaded = false;
        
        // while (!node_stack.empty()) {
        //     auto node = node_stack.top();
        //     node_stack.pop();
            
        //     if (node.get().is_leaf() && node.get().has_active_filter()) {
        //         ID_TYPE node_id = node.get().get_id();
        //         attempted_load_count++;
                
        //         std::string alphas_filepath = config_.get().index_load_folderpath_ + 
        //                                   "filter_" + std::to_string(node_id) + "_alphas.bin";
                
        //         if (fs::exists(alphas_filepath)) {
        //             // printf("加载过滤器 %ld 的批处理alpha值: %s\n", 
        //             //       static_cast<long>(node_id), alphas_filepath.c_str());
                    
        //             // 加载alpha值
        //             RESPONSE load_result = node.get().load_filter_batch_alphas(alphas_filepath);
        //             if (load_result == SUCCESS) {
        //                 loaded_filter_count++;
        //                 // printf("成功加载过滤器 %ld 的批处理alpha值\n", static_cast<long>(node_id));
        //             } else {
        //                 printf("加载过滤器 %ld 的批处理alpha值失败\n", static_cast<long>(node_id));
        //             }
        //         } else {
        //             printf("未找到过滤器 %ld 的批处理alpha文件: %s\n", 
        //                   static_cast<long>(node_id), alphas_filepath.c_str());
        //         }
                
        //         // 加载批处理校准查询ID - 移除打印
        //         std::string calib_query_ids_filepath = config_.get().index_load_folderpath_ + 
        //                                            "filter_" + std::to_string(node_id) + "_calib_query_ids.txt";
                
        //         if (fs::exists(calib_query_ids_filepath)) {
        //             // 静默加载校准查询ID
        //             node.get().load_filter_batch_calib_query_ids(calib_query_ids_filepath);
        //         }

        //         // 检查该过滤器是否有校准查询ID
        //         auto& filter = node.get().get_filter().get();
        //         if (!filter.get_batch_calib_query_ids().empty()) {
        //             any_calib_query_ids_loaded = true;
        //         }
        //     }
            
        //     // 将非叶子节点的子节点加入栈中继续遍历
        //     if (!node.get().is_leaf()) {
        //         for (auto child_node : node.get()) {
        //             node_stack.push(child_node);
        //         }
        //     }
        // }
        
        // printf("尝试加载 %d 个过滤器，成功加载 %d 个过滤器的批处理alpha值\n", 
        //       attempted_load_count, loaded_filter_count);

        // 如果没有成功加载任何校准批次查询ID，则生成合成数据
        if (!any_calib_query_ids_loaded && loaded_filter_count > 0) {
            printf("未加载到任何校准批次查询ID 和 批处理alpha值\n");
        }

        // 5.3 加载预计算的error值（新增）
        if (config_.get().load_precalculated_errors_) {
          std::string error_file_path = config_.get().precalculated_errors_filepath_;
          if (!error_file_path.empty()) {
            printf("\n===== 准备加载预计算的filter error值 =====\n");
            load_filter_errors_from_file(error_file_path);
          } else {
            printf("\n警告: 启用了加载预计算error，但未指定文件路径\n");
          }
        }
      }

      // ==== 第六部分：设置设备 ====
      // 针对训练和推理设置不同设备
      printf("\n===== 设置计算设备 =====\n");
      if (config_.get().filter_infer_is_gpu_){
        printf("为推理设置CUDA设备 %ld\n", (long)config_.get().filter_device_id_);
        device_ = std::make_unique<torch::Device>(torch::kCUDA,
                                               static_cast<c10::DeviceIndex>(config_.get().filter_device_id_));
      } else {
        printf("为推理设置CPU设备\n");
        device_ = std::make_unique<torch::Device>(torch::kCPU);
      }
    }
  }

  // ==== 第七部分：学习导航器设置 ====
  if (config_.get().navigator_is_learned_)
  {
    printf("\n===== 训练导航器 =====\n");
    train();
  }
  
  // 释放缓冲区
  std::free(ifs_buf);
  printf("===== 增强版加载完成 =====\n");
  return SUCCESS;
}





void dstree::Index::init_wrong_pruning_file() {
  if (!wrong_pruning_file_initialized_) {
    std::string save_path = config_.get().save_path_;
    std::string file_path = save_path + "/wrong_pruning_records.csv";
    wrong_pruning_file_.open(file_path);
    
    if (!wrong_pruning_file_.is_open()) {
      printf("错误: 无法创建错误剪枝记录文件 %s\n", file_path.c_str());
      return;
    }
    
    // 写入CSV表头
    wrong_pruning_file_ << "query_id,node_id,true_distance,minbsf,calibrated_distance,predict_raw_distance,true_error,calib_error\n";
    wrong_pruning_file_initialized_ = true;
    
    printf("已创建错误剪枝记录文件: %s\n", file_path.c_str());
  }
}




void dstree::Index::log_pruning_to_csv(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE pred_raw_distance, 
                                       VALUE_TYPE calib_error, VALUE_TYPE predicted_nn_distance, 
                                       VALUE_TYPE minbsf, VALUE_TYPE true_nn_distance) {
    static bool header_written = false;
    std::ofstream csv_file;
    
    // 打开文件，如果不存在则创建，使用追加模式
    // csv_file.open("/home/qwang/projects/leafi/dstree2/result/Recheck_error/pruning_log.csv", std::ios::app);
    std::string save_path = config_.get().save_path_;
    if (!save_path.empty()) {
      namespace fs = boost::filesystem;
      if (!fs::exists(save_path)) {
        fs::create_directories(save_path);
      }
    }
    std::string csv_file_path = save_path + "/pruning_log.csv";
    csv_file.open(csv_file_path, std::ios::app);
    
    // 如果是第一次写入，添加表头
    if (!header_written) {
        csv_file << "query_id,node_id,pred_raw_distance,error_calib,predicted_nn_distance,minbsf,true_error,true_distance\n";
        header_written = true;
    }
    
    // 计算true_error
    VALUE_TYPE true_error = pred_raw_distance - true_nn_distance;
    
    // 写入数据行
    csv_file << query_id << ","
             << node_id << ","
             << pred_raw_distance << "," // 转换为非平方距离
             << calib_error << ","
             << predicted_nn_distance << "," // 转换为非平方距离
             << minbsf << "," // 转换为非平方距离
             << true_error << ","
             << true_nn_distance << "\n";
    
    csv_file.close();
}



// ===============================
// Log helper: record all test distances/errors into CSV
// ===============================
void dstree::Index::log_all_test_to_csv(ID_TYPE query_id, ID_TYPE node_id, VALUE_TYPE pred_raw_distance,
                                        VALUE_TYPE true_nn_distance, VALUE_TYPE predicted_nn_distance,
                                        VALUE_TYPE calib_error, VALUE_TYPE true_error, VALUE_TYPE minbsf) {
  static bool header_written = false;
  std::ofstream csv_file;

  // Build file path under save_path_
  std::string save_path = config_.get().save_path_;
  if (!save_path.empty()) {
    namespace fs = boost::filesystem;
    if (!fs::exists(save_path)) {
      fs::create_directories(save_path);
    }
  }
  std::string csv_file_path = save_path + "/all_test_log.csv";
  csv_file.open(csv_file_path, std::ios::app);
  if (!csv_file.is_open()) {
    printf("[log_all_test_to_csv] Cannot open %s\n", csv_file_path.c_str());
    return;
  }

  if (!header_written) {
    csv_file << "query_id,node_id,pred_raw_distance,true_nn_distance,predicted_nn_distance,calib_error,true_error,minbsf\n";
    header_written = true;
  }

  csv_file << query_id << ","
           << node_id << ","
           << pred_raw_distance << ","
           << true_nn_distance << ","
           << predicted_nn_distance << ","
           << calib_error << ","
           << true_error << ","
           << minbsf << "\n";

  csv_file.close();
}

RESPONSE dstree::Index::load_ground_truth_results(const std::string& filepath, VALUE_TYPE* query_buffer) {
  printf("从文件加载ground truth结果: %s\n", filepath.c_str());
  
  if (!boost::filesystem::exists(filepath)) {
    printf("Ground truth结果文件不存在: %s\n", filepath.c_str());
    return FAILURE;
  }

  std::ifstream ifs(filepath, std::ios::binary);
  if (!ifs.is_open()) {
    printf("错误: 无法打开ground truth结果文件: %s\n", filepath.c_str());
    return FAILURE;
  }

  // 读取文件格式版本
  uint32_t version;
  ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
  if (version != 1) {
    printf("错误: 不支持的文件格式版本: %u\n", version);
    ifs.close();
    return FAILURE;
  }

  // 读取结果数量
  uint32_t num_queries;
  ifs.read(reinterpret_cast<char*>(&num_queries), sizeof(num_queries));
  
  printf("加载%u个查询的ground truth结果\n", num_queries);

  // 清空现有结果并重新分配空间
  ground_truth_answers_.clear();
  ground_truth_answers_.reserve(num_queries);

  // 读取每个查询的结果
  for (uint32_t query_id = 0; query_id < num_queries; ++query_id) {
    // 读取这个查询的答案数量
    uint32_t num_answers;
    ifs.read(reinterpret_cast<char*>(&num_answers), sizeof(num_answers));
    
    // printf("DEBUG LOAD: query_id=%u, num_answers=%u\n", query_id, num_answers);
    
    // 创建新的Answers对象 - 修复：使用正确的构造方式
    auto result_set = std::make_shared<dstree::Answers>(config_.get().n_nearest_neighbor_, query_id);
    
    // 读取每个答案
    std::vector<upcite::Answer> answers;
    for (uint32_t i = 0; i < num_answers; ++i) {
      // 读取保存的query_id
      ID_TYPE saved_query_id;
      ifs.read(reinterpret_cast<char*>(&saved_query_id), sizeof(saved_query_id));
      
      // 读取验证数据
      uint32_t validation_length;
      ifs.read(reinterpret_cast<char*>(&validation_length), sizeof(validation_length));
      
  
      
      std::vector<VALUE_TYPE> saved_validation_data(validation_length);
      ifs.read(reinterpret_cast<char*>(saved_validation_data.data()), sizeof(VALUE_TYPE) * validation_length);
      
      // 验证当前query数据是否与保存的一致
      if (saved_query_id != query_id) {
        printf("警告: 保存的query_id (%u) 与当前query_id (%u) 不匹配\n", saved_query_id, query_id);
        spdlog::warn("保存的query_id ({}) 与当前query_id ({}) 不匹配", saved_query_id, query_id);
      }
      
      // 验证时间序列数据一致性
      VALUE_TYPE *current_series_ptr = query_buffer + config_.get().series_length_ * query_id;
      bool data_consistent = true;
      for (uint32_t j = 0; j < validation_length && j < config_.get().series_length_; ++j) {
        if (std::abs(current_series_ptr[j] - saved_validation_data[j]) > 1e-6) {
          data_consistent = false;
          break;
        }
      }
      
      if (!data_consistent) {
        printf("警告: query_id %u 的时间序列数据与保存时不一致！\n", query_id);
        spdlog::warn("query_id {} 的时间序列数据与保存时不一致！", query_id);
      }
      
      ID_TYPE global_offset, node_id;
      VALUE_TYPE nn_dist;
      
      ifs.read(reinterpret_cast<char*>(&global_offset), sizeof(global_offset));
      ifs.read(reinterpret_cast<char*>(&nn_dist), sizeof(nn_dist));
      ifs.read(reinterpret_cast<char*>(&node_id), sizeof(node_id));
      
      // printf("DEBUG LOAD: query id %u, global_offset=%u, nn_dist=%.6f, node_id=%u\n", 
      //        query_id, global_offset, nn_dist, node_id);
      
      // 修复：使用正确的构造函数
      upcite::Answer answer(nn_dist, node_id, global_offset, query_id);
      answers.push_back(answer);
    }
    
    // 将答案按距离从大到小排序后添加到result_set
    // （因为Answers内部使用最大堆，会保持最小的k个答案）
    std::sort(answers.begin(), answers.end(), [](const upcite::Answer& a, const upcite::Answer& b) {
      return a.nn_dist_ > b.nn_dist_; // 从大到小排序
    });
    
    for (const auto& answer : answers) {
      result_set->push_bsf(answer.nn_dist_, answer.node_id_, answer.global_offset_, answer.query_id_);
    }
    
    ground_truth_answers_.push_back(result_set);
  }

  ifs.close();
  printf("成功加载%zu个查询的ground truth结果\n", ground_truth_answers_.size());
  return SUCCESS;
}


