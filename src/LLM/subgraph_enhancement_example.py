import os
import networkx as nx
import torch
from subgraph_enhancer import SubgraphEnhancer
import json
from typing import Tuple, Dict

def create_sample_subgraph() -> Tuple[nx.Graph, Dict[int, str], Dict[Tuple[int, int], str]]:
    """
    创建一个示例子图
    
    Returns:
        子图、节点类型字典和边类型字典
    """
    # 创建图
    G = nx.Graph()
    
    # 添加节点
    nodes = [(1, "user"), (2, "post"), (3, "user"), (4, "comment"), (5, "user")]
    G.add_nodes_from([n[0] for n in nodes])
    node_types = {n[0]: n[1] for n in nodes}
    
    # 添加边
    edges = [
        (1, 2, "creates"),
        (2, 4, "has"),
        (3, 4, "writes"),
        (4, 5, "mentions"),
        (1, 3, "follows")
    ]
    G.add_edges_from([(e[0], e[1]) for e in edges])
    edge_types = {(e[0], e[1]): e[2] for e in edges}
    
    return G, node_types, edge_types

def main():
    # 初始化子图增强器
    enhancer = SubgraphEnhancer(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-3.5-turbo"
    )
    
    # 创建示例子图
    subgraph, node_types, edge_types = create_sample_subgraph()
    
    # 示例1：分析子图模式
    print("Analyzing subgraph pattern...")
    analysis = enhancer.analyze_subgraph_pattern(subgraph, node_types, edge_types)
    print("\nSubgraph Analysis:")
    print(json.dumps(analysis, indent=2))
    
    # 示例2：增强单个预测
    target_pair = (1, 5)  # 预测用户1和用户5之间是否会形成链接
    original_prediction = 0.6  # 假设原始模型预测的概率是0.6
    
    print("\nEnhancing prediction for target pair:", target_pair)
    enhanced_prediction, prediction_analysis = enhancer.enhance_prediction(
        subgraph, node_types, edge_types, target_pair, original_prediction
    )
    
    print(f"Original prediction: {original_prediction:.3f}")
    print(f"Enhanced prediction: {enhanced_prediction:.3f}")
    print("\nPrediction Analysis:")
    print(json.dumps(prediction_analysis, indent=2))
    
    # 示例3：批量增强预测
    # 创建多个子图
    subgraphs = [subgraph] * 3  # 使用相同的子图作为示例
    node_types_list = [node_types] * 3
    edge_types_list = [edge_types] * 3
    target_pairs = [(1, 5), (2, 4), (3, 5)]  # 不同的目标节点对
    original_predictions = torch.tensor([0.6, 0.7, 0.4])  # 原始预测值
    
    print("\nBatch enhancing predictions...")
    enhanced_predictions, batch_analyses = enhancer.batch_enhance_predictions(
        subgraphs,
        node_types_list,
        edge_types_list,
        target_pairs,
        original_predictions,
        batch_size=2
    )
    
    print("\nBatch Results:")
    for i, (orig, enhanced, analysis) in enumerate(zip(original_predictions, enhanced_predictions, batch_analyses)):
        print(f"\nTarget pair {target_pairs[i]}:")
        print(f"Original prediction: {orig:.3f}")
        print(f"Enhanced prediction: {enhanced:.3f}")
        print("Analysis:")
        print(json.dumps(analysis, indent=2))

if __name__ == "__main__":
    main() 