run_python="${PYTHON:-python}"
run_file="${RUN_FILE:-src/main.py}"

exper_name="bce"
common_args="
    --use_onehot_node_feats
    --use_graph_structure
"



dataset="thgl-forum"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 0 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-github"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 1 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid


dataset="thgl-myket"
nohup $run_python $run_file \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    $common_args \
    --use_gpu 0 \
    --device 2 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

dataset="thgl-software"
nohup $run_python $run_file \
    --exper_name $exper_name \
    --dataset $dataset \
    $common_args \
    --use_gpu 0 \
    --device 3 > run_log/run_${dataset}.log 2>&1 &
echo $! > run_log/run_${dataset}.pid

