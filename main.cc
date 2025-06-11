//
// Created by Qitong Wang on 2022/10/1.
// Copyright (c) 2022 Université Paris Cité. All rights reserved.
//

#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include "config.h"
#include "index.h"

namespace dstree = upcite::dstree;
namespace constant = upcite::constant;



int main(int argc, char *argv[]) {
  printf("[DEBUG] === Program Started ===\n");
  printf("[DEBUG] Command line arguments count argc = %d\n", argc);
  fflush(stdout);

  for (int i = 0; i < argc; ++i) {
    printf("[DEBUG] Argument %d: %s\n", i, argv[i]);
  }
  fflush(stdout);
  // ==================== 1. Initialize Configuration ====================
  printf("[DEBUG] Creating Config object...\n");
  fflush(stdout);
  std::unique_ptr<dstree::Config> config = std::make_unique<dstree::Config>(argc, argv);
  // ==================== 2. Initialize Logger ====================
  printf("[DEBUG] Initializing logger, log path: %s\n", config->log_filepath_.c_str());
  fflush(stdout);
  std::shared_ptr<spdlog::logger> logger = spdlog::basic_logger_mt(constant::LOGGER_NAME, config->log_filepath_);
  
  //==================== Initialize Second Logger ====================
  // std::shared_ptr<spdlog::logger> debug_logger = spdlog::basic_logger_mt("debug_logger", config->log_filepath_qyl_);

#ifdef DEBUG
  logger->set_level(spdlog::level::trace);
  logger->flush_on(spdlog::level::debug);

  //QYL: Configure the second logger for detailed debug logs
  // debug_logger->flush_on(spdlog::level::debug);
#else
  printf("[DEBUG] RELEASE mode: Setting concise log format\n");
  logger->set_pattern("%C-%m-%d %H:%M:%S.%e %L %v");
  logger->set_level(spdlog::level::info);
  logger->flush_on(spdlog::level::err);
#endif
  // spdlog::info("Loaded data shape: {}x{}", num_samples, num_dimensions);
  spdlog::set_default_logger(logger);
  // QYL
  // spdlog::set_default_logger(debug_logger);

  // ==================== 3. Print Configuration Parameters ====================
  printf("[DEBUG] Printing current configuration parameters...\n");
  fflush(stdout);
  config->log(); // Assumes config->log() will print configuration details

  // ==================== 4. Create Index Object ====================
  std::unique_ptr<dstree::Index> index = std::make_unique<dstree::Index>(*config);
  RESPONSE status;

  // ==================== 5. Decide Whether to Load or Build Index ====================
  printf("[DEBUG] Checking if index needs to be loaded: config->to_load_index_ = %d\n", config->to_load_index_);
  fflush(stdout);
  std::string save_path = config->results_path_; // Get path from configuration

  if (config->to_load_index_) {
    printf("\n");
    printf("[DEBUG] Entering index->load() branch\n"); 
    fflush(stdout);  
    // status = index->load();
    auto load_start_time = std::chrono::high_resolution_clock::now();
    status = index->load_enhanced();
    auto load_end_time = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(load_end_time - load_start_time);
    printf("总load_enhanced时间: %ld 微秒\n", load_duration.count());
    printf("\n");

  } else {
    printf("\n");
    printf("[DEBUG] Entering index->build() branch\n");
    fflush(stdout);  
    auto build_start_time = std::chrono::high_resolution_clock::now();
    status = index->build();
    auto build_end_time = std::chrono::high_resolution_clock::now();
    auto build_duration = std::chrono::duration_cast<std::chrono::microseconds>(build_end_time - build_start_time);
    printf("总build时间: %ld 微秒\n", build_duration.count());
    printf("\n");
  }
  
  // ==================== 6. Check If Index Needs to be Dumped ====================
  printf("[DEBUG] Checking if index needs to be dumped: config->to_dump_index_ = %d\n", config->to_dump_index_);
  fflush(stdout);
  if (config->to_dump_index_) {
    printf("\n[DEBUG] Executing index->dump()...\n");
    fflush(stdout);
    auto dump_start_time = std::chrono::high_resolution_clock::now();
    status = static_cast<RESPONSE>(status | index->dump());
    auto dump_end_time = std::chrono::high_resolution_clock::now();
    auto dump_duration = std::chrono::duration_cast<std::chrono::microseconds>(dump_end_time - dump_start_time);
    printf("总dump时间: %ld 微秒\n", dump_duration.count());
  }
  // ========================= 7. Error Handling =========================
  if (status == FAILURE) {
    exit(-1);
  }
  // ========================= 8. Execute Search =========================
  printf("\n[DEBUG] Executing search, profiling mode: config->to_profile_search_ = %d\n", config->to_profile_search_);
  printf("[DEBUG] Entering index->search() branch\n");
  printf("config_.get().filter_remove_square_ = %d\n", config->filter_remove_square_);
  
  

  auto search_total_start_time = std::chrono::high_resolution_clock::now();
  index->search(config->to_profile_search_);
  auto search_total_end_time = std::chrono::high_resolution_clock::now();
  auto search_total_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_total_end_time - search_total_start_time);
  printf("总搜索时间: %ld 微秒\n", search_total_duration.count());
  // index->collect_all_query_prediction_data(config->query_prediction_error_path_);

  // 显式释放index对象，确保有序析构
  printf("[DEBUG] 释放index对象...\n");
  index.reset();

  fflush(stdout);
  return 0;
}
