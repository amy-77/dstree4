实验3: model 1.5, deep1b，load:  q=1k
进程号: 
./dstree2 --db_filepath /mnthdd/data/indexing/deep1b/deep1b-96-25m.bin \
--index_load_folderpath /home/qwang/projects/leafi/dstree2/result/dump_train2k_index_25M_leaf1w_QN2_q10_single \
--results_path /home/qwang/projects/leafi/dstree2/result/train2k_25M_leaf1w_q10_QN2_single \
--save_path /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single \
--series_length 96 \
--db_size 25000000 \
--leaf_size 10000 \
--exact_search \
--require_neurofilter \
--filter_train_is_gpu \
--filter_infer_is_gpu \
--learning_rate 0.01 \
--filter_train_min_lr 0.000001 \
--filter_train_clip_grad \
--filter_train_nepoch 1000 \
--filter_remove_square \
--filter_is_conformal \
--filter_model_setting mlp \
--load_filters \
--load_index \
--load_precalculated_errors \
--filter_trial_nnode 1 \
--filter_allocate_is_gain \
--filter_conformal_is_smoothen \
--filter_train_mthread \
--filter_train_nthread 48 \
--filter_collect_nthread 1 \
--filter_train_num_local_example 500 \
--filter_train_num_global_example 1500 \
--filter_conformal_k_parts 1 \
--filter_conformal_n_parts 12 \
--filter_train_val_split 0.8 \
--filter_query_max_noise 0.1 \
--filter_query_min_noise 0.4 \
--filter_conformal_use_combinatorial \
--filter_conformal_num_batches 100 \
--n_nearest_neighbor 1 \
--query_size 10000 \
--query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin \
--ground_truth_path /home/qwang/projects/leafi/dstree2/deep1b_result_model1.5/ground_truth_25M_leaf1w_q10k_model1.5 \
--filter_conformal_recall 0.90 \
--filter_conformal_coverage 0.90 \
--log_filepath /home/qwang/projects/leafi/dstree2/deep1b_result_model1.5/log_25m_noquantile/load_25M_leaf1w_q10k_QN2_R90_C90_model1.5.log \
--precalculated_errors_filepath /home/qwang/projects/leafi/dstree2/deep1b_result_model1.5/key_points_predictions_without_quantile_1.5/pred_errors_recall0.900_cov0.900.txt \
--device_id 1 




6月9新跑的，

跑2倍增强的model： /mnt/cache/qiyanlin/leafi/lgbm_training_25m_leaf1w_noquantile_2.log




训练模型：2.5倍增强的model
1300840
nohup python /home/qwang/projects/leafi/dstree2/result/python_code/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/lgbm_data_augmented_2.5 --individual --output /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models_individual_2.5 --learning_rate 0.03 --min_coverage 0.5 --max_workers 50 > lgbm_training_25m_leaf1w_noquantile_2.log 2>&1 &

训练模型：基于原始数据直接训练，不增强
nohup python /home/qwang/projects/leafi/dstree2/result/python_code/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/lgbm_data --individual --output /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models_individual_original --learning_rate 0.03 --min_coverage 0.5 --max_workers 40 > lgbm_training_25m_leaf1w_noquantile_original.log 2>&1 &
训练模型：基于原始数据直接训练，不增强 train 200*200
nohup python /home/qwang/projects/leafi/dstree2/result/python_code/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single_BN200_BS200/lgbm_data --individual --output /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models_individual_original_BN200_BN200 --learning_rate 0.03 --min_coverage 0.5 --max_workers 40 > lgbm_training_25m_leaf1w_noquantile_original_BS200_BN200.log 2>&1 &



原始数据train的model 去load error，q=1k


./dstree2 --db_filepath /mnthdd/data/indexing/deep1b/deep1b-96-25m.bin \
--index_load_folderpath /home/qwang/projects/leafi/dstree2/result/dump_train2k_index_25M_leaf1w_QN2_q10_single \
--results_path /home/qwang/projects/leafi/dstree2/result/train2k_25M_leaf1w_q10_QN2_single \
--save_path /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single \
--series_length 96 \
--db_size 25000000 \
--leaf_size 10000 \
--exact_search \
--require_neurofilter \
--filter_train_is_gpu \
--filter_infer_is_gpu \
--learning_rate 0.01 \
--filter_train_min_lr 0.000001 \
--filter_train_clip_grad \
--filter_train_nepoch 1000 \
--filter_remove_square \
--filter_is_conformal \
--filter_model_setting mlp \
--load_filters \
--load_index \
--load_precalculated_errors \
--filter_trial_nnode 1 \
--filter_allocate_is_gain \
--filter_conformal_is_smoothen \
--filter_train_mthread \
--filter_train_nthread 48 \
--filter_collect_nthread 1 \
--filter_train_num_local_example 500 \
--filter_train_num_global_example 1500 \
--filter_conformal_k_parts 1 \
--filter_conformal_n_parts 12 \
--filter_train_val_split 0.8 \
--filter_query_max_noise 0.1 \
--filter_query_min_noise 0.4 \
--filter_conformal_use_combinatorial \
--filter_conformal_num_batches 100 \
--n_nearest_neighbor 1 \
--query_size 1000 \
--query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-1k.bin \
--ground_truth_path /home/qwang/projects/leafi/dstree2/deep1b_result_model_1.25/ground_truth_25M_leaf1w_q1k_model_1.25 \
--filter_conformal_recall 0.99 \
--filter_conformal_coverage 0.99 \
--log_filepath /home/qwang/projects/leafi/dstree2/deep1b_result_model_1.25/log_25m_noquantile/load_25M_leaf1w_q1k_QN2_R99_C99_model_1.25.log \
--precalculated_errors_filepath /home/qwang/projects/leafi/dstree2/deep1b_result_model_1.25/key_points_predictions_without_quantile_1.25/pred_errors_recall0.990_cov0.990.txt \
--device_id 0 


