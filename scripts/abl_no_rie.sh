run_python="${PYTHON:-python}"
run_file="${RUN_FILE:-src/main.py}"

exper_name=$(basename "$0" .sh)
common_args="
    --use_graph_structure
    --use_cached_subgraph
    --model hetero_sthn
"



dataset="thgl-forum"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}_abl.log 2>&1 &
echo $! > run_log/run_${dataset}_abl.pid

# dataset="thgl-github"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid


# dataset="thgl-myket"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid

# dataset="thgl-software"
# nohup $run_python $run_file \
#     --exper_name $exper_name \
#     --dataset $dataset \
#     $common_args \
#     --use_gpu 1 \
#     --device 1 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid


# 测试

# dataset="thgl-software"
# $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 0 \
#     --device 0