#!/bin/bash

# 基础配置
python_exec="${PYTHON:-python}"
main_file="${RUN_FILE:-src/main.py}"
dataset="thgl-forum"

# 统一参数 (保证公平: 相同的 batch_size 和 epoch)
# 我们只跑 5 个 epoch 就足够测出速度和显存了，不需要跑完 100 个
common_args="--dataset $dataset --num_epoch 5 --num_run 1 --batch_size 600 --use_gpu 1 "

echo "🚀 开始效率对比实验 (Efficiency Analysis)..."

# 1. 运行 STHN (Baseline)
echo "Running STHN (Baseline)..."
nohup $python_exec $main_file \
    --exper_name "efficiency_sthn" \
    $common_args \
    --model sthn \
    --use_graph_structure \
    --use_cached_subgraph \
    --device 1 \
    > run_log/efficiency_sthn.log 2>&1 &

    # 注意：不加 --use_riemannian_structure

# 2. 运行 H-SACT (Ours)
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

echo "🎉 效率测试完成！请运行分析脚本查看对比图。"