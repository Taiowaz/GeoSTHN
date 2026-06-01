#!/bin/bash


python_exec="${PYTHON:-python}"
main_file="${RUN_FILE:-src/main.py}"
dataset="thgl-forum"



common_args="--dataset $dataset --num_epoch 5 --num_run 1 --batch_size 600 --use_gpu 1 "

echo "Starting efficiency analysis..."


echo "Running STHN (Baseline)..."
nohup $python_exec $main_file \
    --exper_name "efficiency_sthn" \
    $common_args \
    --model sthn \
    --use_graph_structure \
    --use_cached_subgraph \
    --device 1 \
    > run_log/efficiency_sthn.log 2>&1 &




echo "Running H-SACT (Ours)..."
nohup $python_exec $main_file \
    --exper_name "efficiency_hsact" \
    $common_args \
    --model hetero_sthn \
    --use_graph_structure \
    --use_cached_subgraph \
    --use_riemannian_structure \
    --device 0 \
    > run_log/efficiency_hsact.log 2>&1 &

echo "Efficiency analysis completed. Run the analysis script to view comparison plots."