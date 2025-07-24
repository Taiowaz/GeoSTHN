import pandas as pd
from collections import defaultdict
import torch
from src.structure_enhence.motif import get_rich_edge_motif_strength

META_PATHS_CONFIG = {
    "thgl-github-subset": {
        "URU": [2, 1, 2],
        "UIU": [2, 0, 2],
        "UPRRU": [2, 3, 1, 2],
        "RUR": [1, 2, 1]
    },
    "thgl-software-subset": {
        "URU": [2, 1, 2],
        "UIU": [2, 0, 2],
        "UPRRU": [2, 3, 1, 2],
        "RUR": [1, 2, 1]
    },
    "thgl-myket-subset": {
        "UAU": [0, 1, 0],
        "AUA": [1, 0, 1]
    },
    "thgl-forum-subset": {
        "UBU": [0, 1, 0],
        "BUB": [1, 0, 1]
    }
}

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
) -> torch.Tensor:
    batch_node_features = []
    meta_paths_to_extract = META_PATHS_CONFIG.get(args.dataset, {})
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

    # 4. 将所有节点的特征向量堆叠成一个最终的Tensor
    return torch.tensor(batch_node_features, dtype=torch.float32).to(args.device)



def get_all_node_type_map(dataset_name) -> dict:
    df = pd.read_csv(f"tgb/DATA/{dataset_name.replace('-', '_')}/{dataset_name}_nodetype.csv")
    # 将df的node_id列作为键，type列作为值，转换为字典
    node_type_map = dict(zip(df['node_id'], df['type']))
    return node_type_map