run_python="/home/handb/.conda/envs/geosthn/bin/python"
run_file="/home/handb/GeoSTHN/src/main.py"

exper_name=$(basename "$0" .sh)
common_args="
    --use_onehot_node_feats
    --use_graph_structure
    --use_cached_subgraph
    --use_riemannian_structure
"



# dataset="thgl-forum-subset"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 1 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid

# dataset="thgl-github-subset"
# nohup $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 1 \
#     --device 1 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid


dataset="thgl-myket-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}_abl.log 2>&1 &
echo $! > run_log/run_${dataset}_abl.pid

# dataset="thgl-software-subset"
# nohup $run_python $run_file \
#     --exper_name $exper_name \
#     --dataset $dataset \
#     $common_args \
#     --use_gpu 1 \
#     --device 0 > run_log/run_${dataset}_abl.log 2>&1 &
# echo $! > run_log/run_${dataset}_abl.pid


# 测试

# dataset="thgl-software-subset"
# $run_python $run_file \
#     --exper_name ${exper_name} \
#     --dataset ${dataset} \
#     $common_args \
#     --use_gpu 0 \
#     --device 0