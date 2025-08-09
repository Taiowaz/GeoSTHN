run_python="/private/miniconda3/envs/llm-cdhg/bin/python"
run_file="/private/LLM-CDHG/src/main.py"

exper_name="adaptive-metapath-analysis"
common_args="
    --use_onehot_node_feats
    --use_graph_structure
    --use_motif_metapath_feats
"



dataset="thgl-forum-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 1 \
    --device 0 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-github-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 1 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid


dataset="thgl-myket-subset"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 1 \
    --device 2 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-software-subset"
nohup $run_python $run_file \
    --exper_name $exper_name \
    --dataset $dataset \
    $common_args \
    --use_gpu 1 \
    --device 3 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

