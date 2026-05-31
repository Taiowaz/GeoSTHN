#!/bin/bash

# 基础配置
python_exec="${PYTHON:-python}"
main_file="${RUN_FILE:-src/main.py}"
# dataset="thgl-github"
# dataset="thgl-myket"
dataset="thgl-forum"
common_args="--use_graph_structure --model hetero_sthn --use_cached_subgraph --use_riemannian_structure --use_gpu 0 --num_run 1 --num_epoch 1 --device 1"

target_features=("rgfm_embed_dim" "window_size" "structure_time_gap")

echo "🚀 开始全量参数敏感性分析..."
for feature in "${target_features[@]}"; do
    # 根据特征选择对应的数值列表
    if [ "$feature" == "rgfm_embed_dim" ]; then
        nums=(8 16 32 64 128)
    elif [ "$feature" == "window_size" ]; then
        nums=(2 5 10 25 50)
    elif [ "$feature" == "structure_time_gap" ]; then
        nums=(500 1000 2000 4000 8000)
    fi


    echo "========================================================"
    echo "👉 当前分析特征: ${feature}, 测试数值: [${nums[*]}]"
    echo "========================================================"
    echo "🚀 开始参数敏感性分析：属性${feature}测试数值 [${nums[*]}]"

    for num in "${nums[@]}"; do
        # 为每个维度创建一个独立的实验文件夹
        exper_name="sensitivity_${feature}_${num}"
        mkdir -p "./exper/${exper_name}"  # <--- 修正了这里，从 osmkdir 改为 mkdir

        echo "------------------------------------------------"
        echo "▶️ Running Dimension: ${num} (Experiment: ${exper_name})"
        echo "------------------------------------------------"

        # 这里的命令会阻塞，直到 python 程序运行结束才会进入下一次循环
        $python_exec $main_file \
            --exper_name ${exper_name} \
            --dataset ${dataset} \
            $common_args \
            --${feature} ${num} \

        echo "✅ Dimension ${num} finished."
    done

done

echo "🎉 所有实验运行完毕！"