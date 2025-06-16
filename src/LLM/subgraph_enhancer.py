import os
from typing import Dict, Any, List, Optional, Tuple
from openai import OpenAI
import torch
import networkx as nx
from tqdm import tqdm
import json
import numpy as np


class SubgraphEnhancer:
    """使用LLM分析子图结构来增强预测的类"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        初始化子图增强器

        Args:
            model: 使用的模型名称
        """
        self.client = OpenAI()
        self.model = model

    def _get_llm_analysis(self, prompt: str) -> str:
        """
        使用OpenAI API获取子图分析

        Args:
            prompt: 输入提示

        Returns:
            LLM的分析结果
        """
        try:
            response = self.client.responses.create(model=self.model, input=prompt)
            return response.output_text
        except Exception as e:
            print(f"Error getting LLM analysis: {e}")
            return ""

    def generate_prompt_from_subgraph(self, subgraph_info: Dict[str, Any]) -> str:
        """
        根据子图信息生成提示词

        Args:
            subgraph_info: 子图信息字典

        Returns:
            构建的提示词
        """
        # 提取子图信息
        row = subgraph_info["row"]
        col = subgraph_info["col"]
        eid = subgraph_info["eid"]
        nodes = subgraph_info["nodes"]
        dts = subgraph_info["dts"]
        num_nodes = subgraph_info["num_nodes"]
        num_edges = subgraph_info["num_edges"]
        root_node = subgraph_info["root_node"]
        root_time = subgraph_info["root_time"]

        # 构建子图描述
        description = []
        description.append(
            f"Subgraph centered at root node {root_node} (absolute time: {root_time}):"
        )
        description.append(f"- Number of nodes: {num_nodes}")
        description.append(f"- Number of edges: {num_edges}")
        description.append("\nNodes:")
        for i, node in enumerate(nodes):
            description.append(
                f"  - Node {node} (local index: {i}, time difference: {dts[i]:.2f})"
            )
        description.append("\nEdges:")
        for i in range(len(row)):
            description.append(
                f"  - Edge {nodes[row[i]]} -> {nodes[col[i]]} (edge ID: {eid[i]})"
            )

        # 构建提示词
        prompt = r"""Analyze the following subgraph structure:
    
    {description}
    
    Please analyze:
    1. The overall structure of the subgraph
    2. The temporal relationships between nodes
    3. The connectivity patterns
    4. Any notable structural features
    5. The potential for link formation between nodes
    
    Provide your analysis in JSON format with the following structure:
    {{
        "structure_analysis": "description of the overall structure",
        "temporal_analysis": "description of temporal relationships",
        "connectivity_analysis": "description of connectivity patterns",
        "link_prediction": {{
            "likelihood": "high/medium/low",
            "reasoning": "explanation for the prediction",
            "supporting_patterns": ["list of patterns that support the prediction"]
        }}
    }}"""
        return prompt.format(description="\n".join(description))

    def enhance_subgraph(self, subgraph_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用LLM分析子图并生成增强信息

        Args:
            subgraph_info: 子图信息字典

        Returns:
            LLM分析结果
        """
        # 生成提示词
        prompt = self.generate_prompt_from_subgraph(subgraph_info)

        # 获取LLM分析
        analysis_text = self._get_llm_analysis(prompt)

        return analysis_text


if __name__ == "__main__":
    subgraph_info = {
        "row": np.array([0, 0], dtype=np.int32),
        "col": np.array([1, 2], dtype=np.int32),
        "eid": np.array([81, 71], dtype=np.int32),
        "nodes": np.array([56, 100, 86], dtype=np.int32),
        "dts": np.array([0.0, 128.0, 256.0], dtype=np.float32),
        "num_nodes": 3,
        "num_edges": 2,
        "root_node": 56,
        "root_time": 1704086000.0,
    }

    enhancer = SubgraphEnhancer()
    analysis = enhancer.enhance_subgraph(subgraph_info)
    print(analysis)
