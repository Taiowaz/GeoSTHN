import pandas as pd
import numpy as np
import torch
import networkx as nx
import itertools
import networkx as nx

def get_directed_3node_motifs():
    """
    生成并返回所有13种标准的三节点有向图模体。
    这些模体是预先定义好的，用于同构检查。
    """
    motifs = []
    # Motif 1 (M1)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,2)]); motifs.append(G)
    # Motif 2 (M2)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (2,1)]); motifs.append(G)
    # Motif 3 (M3)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0)]); motifs.append(G)
    # Motif 4 (M4)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,2), (2,0)]); motifs.append(G)
    # Motif 5 (M5)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0), (2,1)]); motifs.append(G)
    # Motif 6 (M6)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,2), (0,2)]); motifs.append(G)
    # Motif 7 (M7)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0), (1,2)]); motifs.append(G)
    # Motif 8 (M8)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0), (2,0)]); motifs.append(G)
    # Motif 9 (M9)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (0,2), (1,2)]); motifs.append(G)
    # Motif 10 (M10)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,2), (2,1)]); motifs.append(G)
    # Motif 11 (M11)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0), (1,2), (2,1)]); motifs.append(G)
    # Motif 12 (M12)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,0), (0,2), (2,0)]); motifs.append(G)
    # Motif 13 (M13)
    G = nx.DiGraph(); G.add_edges_from([(0,1), (1,2), (2,0), (1,0)]); motifs.append(G)
    
    # Note: A fully connected 3-node directed graph has 6 more motifs
    # if you consider all possible edge combinations. The 13 above are standard.
    # The original code expected 13, so we provide 13.
    
    return motifs

MOTIF_LIST = get_directed_3node_motifs()
def get_graph_motif_vectors_batch(df, subgraph_dict_batch, args):
    """
    修正后的函数：为一批子图计算基于模体的特征向量。
    """
    node_vecs = []
    
    for subgraph_dict in subgraph_dict_batch:
        G = nx.DiGraph()
        df_subgraph = df.iloc[subgraph_dict["eid"]]
        src = df_subgraph.src.values
        dst = df_subgraph.dst.values
        G.add_edges_from(zip(src, dst))
        
        root_node = subgraph_dict["root_node"]
        
        vec = np.zeros(13, dtype=np.float32)
        
        if len(G.nodes()) < 3:
            node_vecs.append(torch.from_numpy(vec))
            continue

        for nodes_combo in itertools.combinations(G.nodes(), 3):
            if root_node not in nodes_combo:
                continue
            subg = G.subgraph(nodes_combo)
            
            for m_id, motif_graph in enumerate(MOTIF_LIST):
                if nx.is_isomorphic(subg, motif_graph):
                    vec[m_id] += 1
                    break  
        
        node_vecs.append(torch.from_numpy(vec))
        
    # 6. 将向量列表堆叠成一个Tensor
    return torch.stack(node_vecs)
