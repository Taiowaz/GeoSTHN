import pandas as pd
from collections import defaultdict
import torch
from src.structure_enhence.motif import get_rich_edge_motif_strength

META_PATHS_CONFIG = {
    "thgl-github-subset": {
        # 基于分析结果的高价值元路径（按得分排序选择前5个）
        "RPUP": [1, 3, 2, 3],    # Repo-PR-User-PR (得分最高: 3444.5)
        "PUPR": [3, 2, 3, 1],    # PR-User-PR-Repo (得分: 2505.8)  
        "PUP": [3, 2, 3],        # PR-User-PR (得分: 1833.5)
        "UPRP": [2, 3, 1, 3],    # User-PR-Repo-PR (得分: 1075.4)
    },
    "thgl-software-subset": {
        # 基于分析结果的高价值元路径（按得分排序选择前5个）
        "UPIP": [2, 3, 0, 3],    # User-PR-Issue-PR (得分最高: 1722.2)
        "RIRU": [1, 0, 1, 2],    # Repo-Issue-Repo-User (得分: 1428.8)
        "URIR": [2, 1, 0, 1],    # User-Repo-Issue-Repo (得分: 1259.0)
        "RURI": [1, 2, 1, 0],    # Repo-User-Repo-Issue (得分: 1217.3)
    
    },
    "thgl-myket-subset": {
        # 基于分析结果的高价值元路径（按得分排序选择前4个）
        "RIRI": [1, 0, 1, 0],    # App-User-App-User (得分最高: 23449.8)
        "IRIR": [0, 1, 0, 1],    # User-App-User-App (得分: 20376.9)
        "IRI": [0, 1, 0],        # User-App-User (得分: 5142.4)
        "RIR": [1, 0, 1],        # App-User-App (得分: 2159.5)
    },
    "thgl-forum-subset": {
        # 基于分析结果的高价值元路径（按得分排序选择前5个）
        "IIRI": [0, 0, 1, 0],    # User-User-Post-User (得分最高: 20224.7)
        "IRII": [0, 1, 0, 0],    # User-Post-User-User (得分: 18509.6)  
        "IRI": [0, 1, 0],        # User-Post-User (得分: 5271.5)
        "IIIR": [0, 0, 0, 1],    # User-User-User-Post (得分: 4045.3)
    }
}

# 添加不同配置策略
def get_metapath_config(dataset_name: str, config_type: str = "default") -> dict:
    """
    获取元路径配置
    Args:
        dataset_name: 数据集名称
        config_type: 配置类型 ("minimal", "default", "extended")
    """
    base_config = META_PATHS_CONFIG.get(dataset_name, {})
    
    if config_type == "minimal":
        # 只使用最重要的2-3个路径，减少计算开销
        minimal_configs = {
            "thgl-github-subset": {
                "RPUP": [1, 3, 2, 3],    # 最高得分
                "PUPR": [3, 2, 3, 1],    # 第二高得分
                "PUP": [3, 2, 3],        # 长度较短，计算效率高
            },
            "thgl-software-subset": {
                "UPIP": [2, 3, 0, 3],    # 最高得分
                "RIRU": [1, 0, 1, 2],    # 第二高得分
                "URIR": [2, 1, 0, 1],    # 第三高得分
            },
            "thgl-myket-subset": {
                "RIRI": [1, 0, 1, 0],    # 最高得分
                "IRIR": [0, 1, 0, 1],    # 第二高得分
                "IRI": [0, 1, 0],        # 长度较短，计算效率高
            },
            "thgl-forum-subset": {
                "IIRI": [0, 0, 1, 0],    # 最高得分
                "IRII": [0, 1, 0, 0],    # 第二高得分
                "IRI": [0, 1, 0],        # 长度较短，计算效率高
            }
        }
        return minimal_configs.get(dataset_name, {})
    
    elif config_type == "extended":
        # 使用更多路径（6-8个）
        extended_configs = {
            "thgl-github-subset": {
                **base_config,
                "PRPU": [3, 1, 3, 2],    # PR-Repo-PR-User (得分: 790.5)
                "UPUP": [2, 3, 2, 3],    # User-PR-User-PR (得分: 499.1)
                "PRP": [3, 1, 3],        # PR-Repo-PR (得分: 438.2)
                "RPU": [1, 3, 2],        # Repo-PR-User (得分: 354.2)
            },
            "thgl-software-subset": {
                **base_config,
                "IRUR": [0, 1, 2, 1],    # Issue-Repo-User-Repo (得分: 1044.2)
                "PIPU": [3, 0, 3, 2],    # PR-Issue-PR-User (得分: 995.0)
                "PUPI": [3, 2, 3, 0],    # PR-User-PR-Issue (得分: 843.5)
                "IPUP": [0, 3, 2, 3],    # Issue-PR-User-PR (得分: 671.0)
            },
            "thgl-myket-subset": {
                **base_config,
                "RI": [1, 0],            # App-User (得分: 720.7)
                "IR": [0, 1],            # User-App (得分: 597.7)
            },
            "thgl-forum-subset": {
                **base_config,
                "IIII": [0, 0, 0, 0],    # User-User-User-User (得分: 3732.2)
                "RIII": [1, 0, 0, 0],    # Post-User-User-User (得分: 2764.0)
                "III": [0, 0, 0],        # User-User-User (得分: 1518.4)
                "IIR": [0, 0, 1],        # User-User-Post (得分: 1449.6)
            }
        }
        return extended_configs.get(dataset_name, base_config)
    
    return base_config

def convert_df_to_adj_list(local_subgraph_df: pd.DataFrame) -> defaultdict:
    adj_list = defaultdict(list)
    for _, row in local_subgraph_df.iterrows():
        src, dst, time, label = row['src'], row['dst'], row['time'], row['label']
        adj_list[src].append((dst, time, label))
    return adj_list

def convert_df_to_adj_list_undirected(local_subgraph_df: pd.DataFrame) -> defaultdict:
    """
    修改版：创建无向图的邻接表，允许双向遍历。
    """
    adj_list = defaultdict(list)
    for _, row in local_subgraph_df.iterrows():
        src, dst, time, label = row['src'], row['dst'], row['time'], row['label']
        # 添加正向边
        adj_list[src].append((dst, time, label))
        # 添加反向边，让图的遍历更灵活
        adj_list[dst].append((src, time, label))
    return adj_list

def find_paths_from_root(
    root_node: int,
    meta_path_def: list,
    adj_list: defaultdict,
    node_type_map: dict
) -> list:
    valid_paths = []

    def _dfs_search(current_path: list):
        current_node = current_path[-1]
        current_depth = len(current_path) - 1

        if node_type_map.get(current_node, -1) != meta_path_def[current_depth]:
            return

        # 如果路径已经达到期望长度，说明找到了一条完整路径
        if len(current_path) == len(meta_path_def):
            valid_paths.append(list(current_path))
            return

        # 探索下一步 (与之前逻辑相同)
        next_expected_node_type = meta_path_def[current_depth + 1]
        for neighbor, _, _ in adj_list.get(current_node, []):
            if node_type_map.get(neighbor, -1) == next_expected_node_type and neighbor not in current_path:
                current_path.append(neighbor)
                _dfs_search(current_path)
                current_path.pop()

    if node_type_map.get(root_node, -1) == meta_path_def[0]:
        _dfs_search([root_node])
    
    return valid_paths


def aggregate_features_from_paths_rich(found_paths: list, local_subgraph_df: pd.DataFrame) -> list:
    path_count = len(found_paths)
    
    total_rich_motif_strength = 0
    if path_count > 0:
        for path in found_paths:
            if len(path) >= 2:
                key_edge_u, key_edge_v = path[0], path[1]
                # 调用新的、更丰富的强度计算函数
                strength = get_rich_edge_motif_strength(key_edge_u, key_edge_v, local_subgraph_df)
                total_rich_motif_strength += strength
    
    avg_rich_motif_strength = total_rich_motif_strength / path_count if path_count > 0 else 0
    
    return [float(path_count), avg_rich_motif_strength]

def get_structural_node_features_batch(
    df: pd.DataFrame, 
    subgraph_dict_batch: list, 
    args, 
    metapath_config_type: str = "default"  # 新增参数
) -> torch.Tensor:
    batch_node_features = []
    
    # 使用动态配置
    meta_paths_to_extract = get_metapath_config(args.dataset, metapath_config_type)
    node_type_map = get_all_node_type_map(args.dataset)

    # 1. 直接遍历批次中的每一个节点信息
    for subgraph_dict in subgraph_dict_batch:
        root_node = subgraph_dict['root_node']
        
        # 2. 构建该节点的局部子图
        local_subgraph_df = df.iloc[subgraph_dict['eid']]
        adj_list = convert_df_to_adj_list_undirected(local_subgraph_df)
        
        node_structural_features = []
        # 3. 遍历所有元路径，为该节点计算特征
        if not meta_paths_to_extract:
            node_structural_features = [0.0, 0.0] # 默认特征
        else:
            for path_name, path_def in meta_paths_to_extract.items():
                found_paths = find_paths_from_root(
                    root_node=root_node,
                    meta_path_def=path_def,
                    adj_list=adj_list,
                    node_type_map=node_type_map
                )
                
                features = aggregate_features_from_paths_rich(found_paths, local_subgraph_df)
                
                node_structural_features.extend(features)
        
        batch_node_features.append(node_structural_features)
    res = torch.tensor(batch_node_features, dtype=torch.float32).to(args.device)
    print(f"Batch node features shape: {res.shape}")
    # 4. 将所有节点的特征向量堆叠成一个最终的Tensor
    return res



def get_all_node_type_map(dataset_name) -> dict:
    df = pd.read_csv(f"tgb/DATA/{dataset_name.replace('-', '_')}/{dataset_name}_nodetype.csv")
    # 将df的node_id列作为键，type列作为值，转换为字典
    node_type_map = dict(zip(df['node_id'], df['type']))
    return node_type_map