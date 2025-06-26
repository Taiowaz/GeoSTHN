dataset="thgl-software-subset"
exper_name="sthn-${dataset}"
nohup /root/miniconda3/envs/llm-cdhg/bin/python /root/LLM-CDHG/src/main.py \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    --use_onehot_node_feats \
    --use_graph_structure \
    --use_gpu 1 \
    --device 0 > run_${dataset}.log 2>&1 &
echo $! > run_${dataset}.pid


dataset="thgl-myket-subset"
exper_name="sthn-${dataset}"
nohup /root/miniconda3/envs/llm-cdhg/bin/python /root/LLM-CDHG/src/main.py \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    --use_onehot_node_feats \
    --use_graph_structure \
    --use_gpu 1 \
    --device 1 > run_${dataset}.log 2>&1 &
echo $! > run_${dataset}.pid


dataset="thgl-github-subset"
exper_name="sthn-${dataset}"
nohup /root/miniconda3/envs/llm-cdhg/bin/python /root/LLM-CDHG/src/main.py \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    --use_onehot_node_feats \
    --use_gpu 1 \
    --use_graph_structure \
    --device 2 > run_${dataset}.log 2>&1 &
echo $! > run_${dataset}.pid


dataset="thgl-forum-subset"
exper_name="sthn-${dataset}"
nohup /root/miniconda3/envs/llm-cdhg/bin/python /root/LLM-CDHG/src/main.py \
    --exper_name ${exper_name} \
    --dataset ${dataset} \
    --use_onehot_node_feats \
    --use_graph_structure \
    --use_gpu 1 \
    --device 3 > run_${dataset}.log 2>&1 &
echo $! > run_${dataset}.pid
