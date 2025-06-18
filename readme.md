训练输入参数：
./dstree4 --db_filepath /mnthdd/data/indexing/deep1b-96-25m.bin --query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.4-10.bin --series_length 96 --db_size 25000000 --query_size 10 --leaf_size 10000 --exact_search --require_neurofilter --filter_train_is_gpu --filter_infer_is_gpu --learning_rate 0.01 --filter_train_min_lr 0.000001 --filter_train_clip_grad --filter_train_nepoch 1000 --filter_remove_square --filter_is_conformal --filter_model_setting mlp --device_id 0 --dump_index --filter_trial_nnode 1 --filter_allocate_is_gain --filter_conformal_is_smoothen --filter_train_nthread 48 --filter_train_mthread --filter_collect_nthread 48 --filter_collect_is_mthread --filter_train_num_local_example 500 --filter_train_num_global_example 3000 --filter_conformal_coverage 0.95 --filter_conformal_recall 0.99 --filter_conformal_batch_size 100 --filter_conformal_num_batches 100 --log_filepath /mnthdd/qwang/results/leafi/dstree4/log/results_25M_changeCPdata_totalerror2.log --index_dump_folderpath /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2 --results_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2 --filter_query_max_noise 0.1 --filter_query_min_noise 0.4 --n_nearest_neighbor 1 --filter_conformal_use_combinatorial --save_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2/save_path_train25M --ground_truth_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2/ground_truth_25M_leaf1w_q10

load输入参数：
load 过程
./dstree4 --db_filepath /mnthdd/data/indexing/deep1b-96-25m.bin \
--index_load_folderpath /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2 \
--results_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2 \
--save_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2/save_path_train25M \
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
--load_index \
--load_filters \
--load_precalculated_errors \
--filter_trial_nnode 1 \
--filter_allocate_is_gain \
--filter_conformal_is_smoothen \
--filter_train_nthread 48 \
--filter_train_mthread \
--filter_collect_nthread 48 \
--filter_collect_is_mthread \
--filter_train_num_local_example 500 \
--filter_train_num_global_example 5000 \
--filter_query_max_noise 0.1 \
--filter_query_min_noise 0.4 \
--filter_conformal_use_combinatorial \
--filter_conformal_batch_size 100 \
--filter_conformal_num_batches 100  \
--n_nearest_neighbor 1 \
--query_size 10000 \
--device_id 0 \
--ground_truth_path /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2/load_totalerror2_q10k\ground_truth_25M_leaf1w_q10k \
--query_filepath /home/qwang/projects/leafi/dataset/deep1b-96-10m-test-0.2-10k.bin \
--log_filepath /mnthdd/qwang/results/leafi/dstree4/load_totalerror2_q10k/load_totalerror2_R90_C90_10k.log \
--filter_conformal_recall 0.9 \
--filter_conformal_coverage 0.9 \
--precalculated_errors_filepath /mnthdd/qwang/results/leafi/dstree4/results_25M_changeCPdata_totalerror2/save_path_train25M/key_points_predictions/pred_errors_recall0.900_cov0.900.txt 



训练模型：基于原始数据直接训练，不增强
nohup python /home/qwang/projects/leafi/dstree2/result/python_code/train_eval_lgbm.py --data_dir /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/lgbm_data --individual --output /home/qwang/projects/leafi/dstree2/result/save_path_train2k_25M_leaf1w_q10_single/quantile_models_individual_original --learning_rate 0.03 --min_coverage 0.5 --max_workers 40 > lgbm_training_25m_leaf1w_noquantile_original.log 2>&1 &




