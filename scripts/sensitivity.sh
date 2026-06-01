#!/bin/bash


python_exec="${PYTHON:-python}"
main_file="${RUN_FILE:-src/main.py}"
# dataset="thgl-github"
# dataset="thgl-myket"
dataset="thgl-forum"
common_args="--use_graph_structure --model hetero_sthn --use_cached_subgraph --use_riemannian_structure --use_gpu 0 --num_run 1 --num_epoch 1 --device 1"

target_features=("rgfm_embed_dim" "window_size" "structure_time_gap")

echo "Starting full parameter sensitivity analysis..."
for feature in "${target_features[@]}"; do

    if [ "$feature" == "rgfm_embed_dim" ]; then
        nums=(8 16 32 64 128)
    elif [ "$feature" == "window_size" ]; then
        nums=(2 5 10 25 50)
    elif [ "$feature" == "structure_time_gap" ]; then
        nums=(500 1000 2000 4000 8000)
    fi


    echo "========================================================"
    echo "Current feature: ${feature}; values: [${nums[*]}]"
    echo "========================================================"
    echo "Starting sensitivity analysis for ${feature}; values: [${nums[*]}]"

    for num in "${nums[@]}"; do

        exper_name="sensitivity_${feature}_${num}"
        mkdir -p "./exper/${exper_name}"

        echo "------------------------------------------------"
        echo "▶️ Running Dimension: ${num} (Experiment: ${exper_name})"
        echo "------------------------------------------------"


        $python_exec $main_file \
            --exper_name ${exper_name} \
            --dataset ${dataset} \
            $common_args \
            --${feature} ${num} \

        echo "✅ Dimension ${num} finished."
    done

done

echo "All experiments completed."