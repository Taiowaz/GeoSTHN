run_python="/private/miniconda3/envs/llm-cdhg/bin/python"
run_file="/private/LLM-CDHG/src/main.py"

exper_name=$(basename "$0" .sh)
common_args="
    --use_onehot_node_feats
    --use_graph_structure
    --model hetero_sthn
    --use_riemannian_structure
"



dataset="thgl-forum-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 0 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-github-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid


dataset="thgl-myket-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 2 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-software-subset"
nohup $run_python $run_file \
    --exper_name $exper_name \
    --dataset $dataset \
    $common_args \
    --use_gpu 0 \
    --device 3 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

