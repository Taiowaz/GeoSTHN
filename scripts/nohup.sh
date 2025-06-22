dataset="thgl-software-subset"
exper_name="sthn-${dataset}-node_feat_enhence-node_fusion"
nohup /root/miniconda3/envs/llm-cdhg/bin/python /root/LLM-CDHG/src/main.py --exper_name ${exper_name} --dataset ${dataset} > run.log 2>&1 &
echo $! > run.pid