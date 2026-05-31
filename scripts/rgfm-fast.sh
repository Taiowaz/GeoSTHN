run_python="${PYTHON:-python}"
run_file="${RUN_FILE:-src/main.py}"

exper_name=$(basename "$0" .sh)
common_args="
    --use_onehot_node_feats
    --use_graph_structure
    --model hetero_sthn
    --use_riemannian_structure
    --use_cached_subgraph
    --batch_size 70
"
# 禁用onehot特征,禁用use_cached_subgraph
# common_args="
#     --use_graph_structure
#     --model hetero_sthn
#     --use_riemannian_structure
#     --batch_size 100
# "



# dataset="thgl-forum"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_fast.log 2>&1 &
# echo $! > run_log/run_${dataset}_fast.pid

# dataset="thgl-github"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_fast.log 2>&1 &
# echo $! > run_log/run_${dataset}_fast.pid


# dataset="thgl-myket"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_fast.log 2>&1 &
# echo $! > run_log/run_${dataset}_fast.pid

dataset="thgl-software"
nohup $run_python $run_file \
    --exper_name $exper_name \
    --dataset $dataset \
    $common_args \
    --use_gpu 1 \
    --device 0 > run_log/run_${dataset}_fast.log 2>&1 &
echo $! > run_log/run_${dataset}_fast.pid


# 测试

# dataset="thgl-software"
# $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0