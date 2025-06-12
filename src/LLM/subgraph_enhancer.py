import os
from typing import Dict, Any, List, Optional, Tuple
import openai
import torch
import networkx as nx
from tqdm import tqdm
import json

class SubgraphEnhancer:
    """使用LLM分析子图结构来增强预测的类"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        初始化子图增强器
        
        Args:
            api_key: OpenAI API密钥，如果为None则从环境变量获取
            model: 使用的模型名称
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.model = model
        openai.api_key = self.api_key
    
    def _get_llm_analysis(self, prompt: str) -> str:
        """
        使用OpenAI API获取子图分析
        
        Args:
            prompt: 输入提示
            
        Returns:
            LLM的分析结果
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3  # 使用较低的温度以获得更稳定的输出
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting LLM analysis: {e}")
            return ""
    
    def _convert_subgraph_to_text(self, 
                                subgraph: nx.Graph,
                                node_types: Dict[int, str],
                                edge_types: Dict[Tuple[int, int], str]) -> str:
        """
        将子图转换为文本描述
        
        Args:
            subgraph: NetworkX图对象
            node_types: 节点类型字典
            edge_types: 边类型字典
            
        Returns:
            子图的文本描述
        """
        description = []
        
        # 添加节点信息
        description.append("Nodes in the subgraph:")
        for node in subgraph.nodes():
            node_type = node_types.get(node, "unknown")
            description.append(f"- Node {node} (Type: {node_type})")
        
        # 添加边信息
        description.append("\nEdges in the subgraph:")
        for u, v in subgraph.edges():
            edge_type = edge_types.get((u, v), "unknown")
            description.append(f"- Edge {u} -> {v} (Type: {edge_type})")
        
        # 添加图的基本统计信息
        description.append("\nSubgraph statistics:")
        description.append(f"- Number of nodes: {subgraph.number_of_nodes()}")
        description.append(f"- Number of edges: {subgraph.number_of_edges()}")
        description.append(f"- Average degree: {sum(dict(subgraph.degree()).values()) / subgraph.number_of_nodes():.2f}")
        
        return "\n".join(description)
    
    def analyze_subgraph_pattern(self,
                               subgraph: nx.Graph,
                               node_types: Dict[int, str],
                               edge_types: Dict[Tuple[int, int], str],
                               target_pair: Optional[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        分析子图模式
        
        Args:
            subgraph: NetworkX图对象
            node_types: 节点类型字典
            edge_types: 边类型字典
            target_pair: 目标节点对（可选）
            
        Returns:
            分析结果字典
        """
        # 将子图转换为文本描述
        subgraph_text = self._convert_subgraph_to_text(subgraph, node_types, edge_types)
        
        # 构建提示
        prompt = f"""Given the following subgraph structure, analyze the patterns and relationships:

{subgraph_text}

Please analyze:
1. The overall structure of the subgraph
2. The types of nodes and their relationships
3. The connectivity patterns
4. Any notable structural features
5. The potential for link formation between nodes

If a target node pair is specified, focus on analyzing the likelihood of a link forming between them.

Please provide your analysis in JSON format with the following structure:
{{
    "structure_analysis": "description of the overall structure",
    "node_patterns": "description of node type patterns",
    "edge_patterns": "description of edge type patterns",
    "connectivity_analysis": "description of connectivity patterns",
    "link_prediction": {{
        "likelihood": "high/medium/low",
        "reasoning": "explanation for the prediction",
        "supporting_patterns": ["list of patterns that support the prediction"]
    }}
}}"""

        # 获取LLM分析
        analysis_text = self._get_llm_analysis(prompt)
        
        try:
            # 解析JSON响应
            analysis = json.loads(analysis_text)
            return analysis
        except:
            # 如果解析失败，返回空字典
            return {}
    
    def enhance_prediction(self,
                         subgraph: nx.Graph,
                         node_types: Dict[int, str],
                         edge_types: Dict[Tuple[int, int], str],
                         target_pair: Tuple[int, int],
                         original_prediction: float) -> Tuple[float, Dict[str, Any]]:
        """
        使用子图分析增强预测
        
        Args:
            subgraph: NetworkX图对象
            node_types: 节点类型字典
            edge_types: 边类型字典
            target_pair: 目标节点对
            original_prediction: 原始预测值
            
        Returns:
            增强后的预测值和分析结果
        """
        # 获取子图分析
        analysis = self.analyze_subgraph_pattern(subgraph, node_types, edge_types, target_pair)
        
        # 根据分析结果调整预测值
        if "link_prediction" in analysis:
            likelihood = analysis["link_prediction"]["likelihood"].lower()
            if likelihood == "high":
                adjustment = 0.2
            elif likelihood == "medium":
                adjustment = 0.0
            else:  # low
                adjustment = -0.2
        else:
            adjustment = 0.0
        
        # 应用调整
        enhanced_prediction = max(0.0, min(1.0, original_prediction + adjustment))
        
        return enhanced_prediction, analysis
    
    def batch_enhance_predictions(self,
                                subgraphs: List[nx.Graph],
                                node_types_list: List[Dict[int, str]],
                                edge_types_list: List[Dict[Tuple[int, int], str]],
                                target_pairs: List[Tuple[int, int]],
                                original_predictions: torch.Tensor,
                                batch_size: int = 32) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        批量增强预测
        
        Args:
            subgraphs: 子图列表
            node_types_list: 节点类型字典列表
            edge_types_list: 边类型字典列表
            target_pairs: 目标节点对列表
            original_predictions: 原始预测值张量
            batch_size: 批处理大小
            
        Returns:
            增强后的预测值张量和分析结果列表
        """
        enhanced_predictions = []
        all_analyses = []
        
        for i in tqdm(range(0, len(subgraphs), batch_size)):
            batch_subgraphs = subgraphs[i:i + batch_size]
            batch_node_types = node_types_list[i:i + batch_size]
            batch_edge_types = edge_types_list[i:i + batch_size]
            batch_target_pairs = target_pairs[i:i + batch_size]
            batch_original = original_predictions[i:i + batch_size]
            
            for subgraph, node_types, edge_types, target_pair, original_pred in zip(
                batch_subgraphs, batch_node_types, batch_edge_types,
                batch_target_pairs, batch_original
            ):
                enhanced_pred, analysis = self.enhance_prediction(
                    subgraph, node_types, edge_types, target_pair, original_pred.item()
                )
                enhanced_predictions.append(enhanced_pred)
                all_analyses.append(analysis)
        
        return torch.tensor(enhanced_predictions), all_analyses 