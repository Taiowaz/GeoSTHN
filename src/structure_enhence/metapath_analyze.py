import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import logging
import json
from pathlib import Path

class MetaPathAnalyzer:
    def __init__(self, dataset_name: str, max_path_length: int = 4):
        """
        元路径分析器
        Args:
            dataset_name: 数据集名称
            max_path_length: 最大路径长度
        """
        self.dataset_name = dataset_name
        self.max_path_length = max_path_length
        
        # 先设置日志，再初始化其他属性
        self._setup_logger()
        
        self.node_type_map = self._load_node_type_map()
        self.edge_data = self._load_edge_data()
        self.type_names = self._get_type_names()
        
    def _setup_logger(self):
        """设置日志"""
        # 创建logger
        self.logger = logging.getLogger(f"{__name__}_{self.dataset_name}")
        self.logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 创建控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            
            # 添加handler到logger
            self.logger.addHandler(console_handler)
        
    def _load_node_type_map(self) -> Dict[int, int]:
        """加载节点类型映射"""
        try:
            file_path = f"tgb/DATA/{self.dataset_name.replace('-', '_')}/{self.dataset_name}_nodetype.csv"
            df = pd.read_csv(file_path)
            self.logger.info(f"加载节点类型文件: {file_path}")
            self.logger.info(f"节点数量: {len(df)}")
            return dict(zip(df['node_id'], df['type']))
        except Exception as e:
            self.logger.error(f"无法加载节点类型文件: {e}")
            return {}
    
    def _load_edge_data(self) -> pd.DataFrame:
        """加载边数据"""
        try:
            file_path = f"tgb/DATA/{self.dataset_name.replace('-', '_')}/{self.dataset_name}_edgelist.csv"
            df = pd.read_csv(file_path)
            self.logger.info(f"加载边数据文件: {file_path}")
            self.logger.info(f"边数量: {len(df)}")
            # 根据实际列名调整
            if 'head' in df.columns and 'tail' in df.columns:
                df = df.rename(columns={'head': 'src', 'tail': 'dst'})
            return df
        except Exception as e:
            self.logger.error(f"无法加载边数据文件: {e}")
            return pd.DataFrame()
    
    def _get_type_names(self) -> Dict[int, str]:
        """获取类型名称映射"""
        type_mapping = {
            "thgl-github-subset": {0: "Issue", 1: "Repo", 2: "User", 3: "PR"},
            "thgl-software-subset": {0: "Issue", 1: "Repo", 2: "User", 3: "PR"},
            "thgl-myket-subset": {0: "User", 1: "App"},
            "thgl-forum-subset": {0: "User", 1: "Post"}
        }
        return type_mapping.get(self.dataset_name, {})
    
    def analyze_dataset_structure(self) -> Dict:
        """分析数据集的基本结构"""
        self.logger.info("开始分析数据集结构...")
        
        if not self.node_type_map:
            self.logger.warning("节点类型映射为空，跳过结构分析")
            return {'dataset': self.dataset_name, 'error': 'No node type mapping'}
        
        if self.edge_data.empty:
            self.logger.warning("边数据为空，跳过结构分析")
            return {'dataset': self.dataset_name, 'error': 'No edge data'}
        
        # 节点类型统计
        type_counts = Counter(self.node_type_map.values())
        total_nodes = len(self.node_type_map)
        
        # 边类型统计
        edge_type_counts = defaultdict(int)
        valid_edges = 0
        
        for _, row in self.edge_data.iterrows():
            src_type = self.node_type_map.get(row['src'], -1)
            dst_type = self.node_type_map.get(row['dst'], -1)
            if src_type != -1 and dst_type != -1:
                edge_type = (src_type, dst_type)
                edge_type_counts[edge_type] += 1
                valid_edges += 1
        
        structure_info = {
            'dataset': self.dataset_name,
            'total_nodes': total_nodes,
            'total_edges': len(self.edge_data),
            'valid_edges': valid_edges,
            'node_types': dict(type_counts),
            'edge_types': dict(edge_type_counts),
            'type_names': self.type_names
        }
        
        self._print_structure_info(structure_info)
        return structure_info
    
    def _print_structure_info(self, info: Dict):
        """打印结构信息"""
        if 'error' in info:
            print(f"\n数据集 {info['dataset']} 分析失败: {info['error']}")
            return
            
        print("\n" + "="*60)
        print(f"数据集结构分析: {info['dataset']}")
        print("="*60)
        
        print(f"总节点数: {info['total_nodes']:,}")
        print(f"总边数: {info['total_edges']:,}")
        print(f"有效边数: {info['valid_edges']:,}")
        
        print("\n节点类型分布:")
        for type_id, count in info['node_types'].items():
            type_name = info['type_names'].get(type_id, f"Type{type_id}")
            ratio = count / info['total_nodes'] * 100
            print(f"  {type_name} (类型{type_id}): {count:,} ({ratio:.1f}%)")
        
        print("\n边类型分布:")
        total_edges = sum(info['edge_types'].values())
        if total_edges > 0:
            for (src_type, dst_type), count in sorted(info['edge_types'].items(), key=lambda x: x[1], reverse=True):
                src_name = info['type_names'].get(src_type, f"Type{src_type}")
                dst_name = info['type_names'].get(dst_type, f"Type{dst_type}")
                ratio = count / total_edges * 100
                print(f"  {src_name} -> {dst_name}: {count:,} ({ratio:.1f}%)")
        else:
            print("  无有效边类型")
    
    def discover_metapaths(self, sample_size: int = 10000, min_support: int = 5) -> List[Dict]:
        """发现元路径"""
        self.logger.info(f"开始发现元路径，采样大小: {sample_size}")
        
        if self.edge_data.empty or not self.node_type_map:
            self.logger.warning("数据不足，无法发现元路径")
            return []
        
        # 采样边数据
        actual_sample_size = min(sample_size, len(self.edge_data))
        sampled_edges = self.edge_data.sample(n=actual_sample_size, random_state=42)
        self.logger.info(f"实际采样边数: {actual_sample_size}")
        
        # 构建图
        graph = self._build_graph(sampled_edges)
        
        if not graph:
            self.logger.warning("构建的图为空")
            return []
        
        # 发现不同长度的元路径
        all_metapaths = []
        
        for length in range(2, self.max_path_length + 1):
            self.logger.info(f"发现长度为 {length} 的元路径...")
            metapaths = self._find_metapaths_by_length(graph, length, min_support)
            self.logger.info(f"长度为 {length} 的元路径数量: {len(metapaths)}")
            all_metapaths.extend(metapaths)
        
        # 按支持度排序
        all_metapaths.sort(key=lambda x: x['support'], reverse=True)
        
        return all_metapaths
    
    def _build_graph(self, edge_data: pd.DataFrame) -> defaultdict:
        """构建图的邻接表"""
        graph = defaultdict(list)
        edge_count = 0
        
        for _, row in edge_data.iterrows():
            src, dst = row['src'], row['dst']
            src_type = self.node_type_map.get(src, -1)
            dst_type = self.node_type_map.get(dst, -1)
            
            if src_type != -1 and dst_type != -1:
                graph[src].append((dst, src_type, dst_type))
                # 添加反向边（无向图）
                graph[dst].append((src, dst_type, src_type))
                edge_count += 1
        
        self.logger.info(f"构建图完成，有效边数: {edge_count}, 节点数: {len(graph)}")
        return graph
    
    def _find_metapaths_by_length(self, graph: defaultdict, length: int, min_support: int) -> List[Dict]:
        """找到指定长度的元路径"""
        metapath_instances = defaultdict(list)
        
        # 限制搜索的起始节点数量
        start_nodes = list(graph.keys())[:min(1000, len(graph))]
        self.logger.info(f"搜索起始节点数: {len(start_nodes)}")
        
        processed_nodes = 0
        for start_node in start_nodes:
            start_type = self.node_type_map.get(start_node, -1)
            if start_type == -1:
                continue
                
            # DFS搜索
            self._dfs_metapath_search(
                graph, start_node, [start_type], [start_node], 
                length, metapath_instances
            )
            
            processed_nodes += 1
            if processed_nodes % 100 == 0:
                self.logger.info(f"已处理 {processed_nodes}/{len(start_nodes)} 个起始节点")
        
        # 统计并过滤
        metapaths = []
        for pattern, instances in metapath_instances.items():
            if len(instances) >= min_support:
                metapaths.append({
                    'pattern': list(pattern),
                    'pattern_str': self._pattern_to_string(pattern),
                    'support': len(instances),
                    'unique_nodes': len(set(node for instance in instances for node in instance)),
                    'avg_length': length,
                    'instances': instances[:10]  # 只保留前10个实例
                })
        
        return metapaths
    
    def _dfs_metapath_search(self, graph: defaultdict, current_node: int, 
                           type_path: List[int], node_path: List[int], 
                           target_length: int, metapath_instances: defaultdict):
        """DFS搜索元路径"""
        if len(type_path) == target_length:
            pattern = tuple(type_path)
            metapath_instances[pattern].append(node_path.copy())
            return
        
        if len(type_path) >= target_length:
            return
        
        # 限制搜索深度和广度
        neighbors = graph.get(current_node, [])[:min(20, len(graph.get(current_node, [])))]
        
        for neighbor, _, neighbor_type in neighbors:
            if neighbor not in node_path:  # 避免循环
                type_path.append(neighbor_type)
                node_path.append(neighbor)
                
                self._dfs_metapath_search(
                    graph, neighbor, type_path, node_path, 
                    target_length, metapath_instances
                )
                
                type_path.pop()
                node_path.pop()
    
    def _pattern_to_string(self, pattern: Tuple[int]) -> str:
        """将模式转换为可读字符串"""
        type_chars = {0: 'I', 1: 'R', 2: 'U', 3: 'P', 4: 'A', 5: 'B'}
        return ''.join(type_chars.get(t, f'T{t}') for t in pattern)
    
    def analyze_metapath_quality(self, metapaths: List[Dict]) -> List[Dict]:
        """分析元路径质量"""
        if not metapaths:
            self.logger.warning("没有元路径需要分析质量")
            return []
            
        self.logger.info("分析元路径质量...")
        
        for metapath in metapaths:
            pattern = metapath['pattern']
            
            # 计算语义合理性
            semantic_score = self._compute_semantic_score(pattern)
            
            # 计算多样性
            diversity_score = self._compute_diversity_score(metapath)
            
            # 计算稀有性
            rarity_score = self._compute_rarity_score(pattern)
            
            # 综合得分
            total_score = (
                metapath['support'] * 0.3 +
                semantic_score * 0.3 +
                diversity_score * 0.2 +
                rarity_score * 0.2
            )
            
            metapath.update({
                'semantic_score': semantic_score,
                'diversity_score': diversity_score,
                'rarity_score': rarity_score,
                'total_score': total_score
            })
        
        # 按综合得分排序
        metapaths.sort(key=lambda x: x['total_score'], reverse=True)
        return metapaths
    
    def _compute_semantic_score(self, pattern: List[int]) -> float:
        """计算语义合理性得分"""
        # 基于领域知识的语义评分
        semantic_patterns = {
            "thgl-github-subset": {
                (2, 1, 2): 10,  # User-Repo-User (协作关系)
                (2, 0, 2): 8,   # User-Issue-User (Issue讨论)
                (2, 3, 1, 2): 9, # User-PR-Repo-User (PR贡献)
                (2, 1): 6,      # User-Repo (简单关系)
                (2, 0): 5,      # User-Issue
                (2, 3): 7,      # User-PR
            },
            "thgl-myket-subset": {
                (0, 1, 0): 10,  # User-App-User (应用使用)
                (0, 1): 8,      # User-App
            },
            "thgl-forum-subset": {
                (0, 1, 0): 10,  # User-Post-User (论坛讨论)
                (0, 1): 8,      # User-Post
            }
        }
        
        dataset_patterns = semantic_patterns.get(self.dataset_name, {})
        return dataset_patterns.get(tuple(pattern), 1.0)
    
    def _compute_diversity_score(self, metapath: Dict) -> float:
        """计算多样性得分"""
        unique_nodes = metapath['unique_nodes']
        support = metapath['support']
        return unique_nodes / support if support > 0 else 0
    
    def _compute_rarity_score(self, pattern: List[int]) -> float:
        """计算稀有性得分"""
        if not self.node_type_map:
            return 1.0
            
        # 计算类型转移的稀有性
        type_counts = Counter(self.node_type_map.values())
        total_nodes = len(self.node_type_map)
        
        rarity = 1.0
        for i in range(len(pattern) - 1):
            from_type, to_type = pattern[i], pattern[i + 1]
            from_freq = type_counts.get(from_type, 1) / total_nodes
            to_freq = type_counts.get(to_type, 1) / total_nodes
            
            # 稀有类型转移得分更高
            transition_rarity = 1.0 / (from_freq * to_freq + 0.001)
            rarity *= transition_rarity
        
        return min(rarity, 100)  # 限制最大值
    
    def generate_config_recommendations(self, metapaths: List[Dict], top_k: int = 10) -> Dict:
        """生成配置推荐"""
        if not metapaths:
            self.logger.warning("没有元路径可以生成推荐")
            return {}
            
        self.logger.info(f"生成前 {min(top_k, len(metapaths))} 个元路径的配置推荐...")
        
        recommendations = {}
        selected_metapaths = metapaths[:top_k]
        
        for i, metapath in enumerate(selected_metapaths):
            pattern_name = metapath['pattern_str']
            pattern = metapath['pattern']
            
            recommendations[pattern_name] = {
                'pattern': pattern,
                'support': metapath['support'],
                'total_score': metapath.get('total_score', 0),
                'description': self._generate_description(pattern)
            }
        
        return recommendations
    
    def _generate_description(self, pattern: List[int]) -> str:
        """生成模式描述"""
        type_names = self.type_names
        if not type_names:
            return f"Pattern: {pattern}"
        
        description_parts = []
        for i, type_id in enumerate(pattern):
            type_name = type_names.get(type_id, f"Type{type_id}")
            if i == 0:
                description_parts.append(type_name)
            else:
                description_parts.append(f"-{type_name}")
        
        return ''.join(description_parts)
    
    def save_analysis_results(self, metapaths: List[Dict], output_dir: str = "metapath_analysis"):
        """保存分析结果"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 保存详细结果
        results_file = output_path / f"{self.dataset_name}_metapath_analysis.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(metapaths, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存配置推荐
        recommendations = self.generate_config_recommendations(metapaths)
        config_file = output_path / f"{self.dataset_name}_metapath_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        # 生成可读报告
        self._generate_readable_report(metapaths, output_path)
        
        self.logger.info(f"分析结果已保存到: {output_path}")
    
    def _generate_readable_report(self, metapaths: List[Dict], output_path: Path):
        """生成可读报告"""
        report_file = output_path / f"{self.dataset_name}_metapath_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"元路径分析报告 - {self.dataset_name}\n")
            f.write("=" * 60 + "\n\n")
            
            if not metapaths:
                f.write("未发现任何有价值的元路径\n")
                return
            
            f.write("推荐的元路径配置:\n")
            f.write("-" * 30 + "\n")
            
            for i, metapath in enumerate(metapaths[:15]):  # 显示前15个
                f.write(f"\n{i+1}. {metapath['pattern_str']}: {metapath['pattern']}\n")
                f.write(f"   描述: {metapath.get('description', self._generate_description(metapath['pattern']))}\n")
                f.write(f"   支持度: {metapath['support']}\n")
                f.write(f"   综合得分: {metapath.get('total_score', 0):.2f}\n")
                f.write(f"   语义得分: {metapath.get('semantic_score', 0):.2f}\n")
                f.write(f"   多样性得分: {metapath.get('diversity_score', 0):.2f}\n")
                f.write(f"   稀有性得分: {metapath.get('rarity_score', 0):.2f}\n")


def run_metapath_analysis(dataset_name: str):
    """运行元路径分析的主函数"""
    print(f"\n开始分析数据集: {dataset_name}")
    print("=" * 60)
    
    try:
        analyzer = MetaPathAnalyzer(dataset_name, max_path_length=4)
        
        # 1. 分析数据集结构
        structure_info = analyzer.analyze_dataset_structure()
        
        if 'error' in structure_info:
            print(f"数据集 {dataset_name} 结构分析失败，跳过元路径发现")
            return
        
        # 2. 发现元路径
        metapaths = analyzer.discover_metapaths(sample_size=20000, min_support=3)
        
        if not metapaths:
            print("未发现任何元路径!")
            return
        
        print(f"\n发现 {len(metapaths)} 个元路径")
        
        # 3. 分析质量
        metapaths = analyzer.analyze_metapath_quality(metapaths)
        
        # 4. 显示推荐
        recommendations = analyzer.generate_config_recommendations(metapaths, top_k=10)
        
        if recommendations:
            print("\n推荐的元路径配置:")
            print("-" * 40)
            for name, info in recommendations.items():
                print(f"{name}: {info['pattern']} (支持度: {info['support']}, 得分: {info['total_score']:.2f})")
                print(f"  描述: {info['description']}")
                print()
        
        # 5. 保存结果
        analyzer.save_analysis_results(metapaths)
        
        print(f"分析完成! 结果已保存到 metapath_analysis/ 目录")
        
    except Exception as e:
        print(f"分析数据集 {dataset_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 分析所有数据集
    datasets = [
        "thgl-github-subset",
        "thgl-software-subset", 
        "thgl-myket-subset",
        "thgl-forum-subset"
    ]
    
    for dataset in datasets:
        run_metapath_analysis(dataset)
        print("\n" + "="*80 + "\n")