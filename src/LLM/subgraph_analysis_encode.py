import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from tqdm import tqdm  # 导入 tqdm 库


class SubgraphEncoder:
    """
    一个将LLM生成的子图分析JSON编码为单个增强向量的编码器。
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化编码器。
        - 加载句子嵌入模型。
        - 定义分类特征的固定词汇表，以确保编码一致性。

        Args:
            model_name (str): 要使用的Sentence Transformer模型名称。
        """
        # 1. 加载文本嵌入模型 (此操作较慢，应在初始化时完成)
        print("Loading Sentence Transformer model...")
        self.st_model = SentenceTransformer(model_name)
        print("Model loaded.")

        # 2. 为分类特征定义固定的类别词汇表
        # 添加 "Unknown" 以处理未来可能出现的新类别
        self.HEALTH_CATEGORIES = ["Healthy", "Problematic", "Concerning", "Unknown"]
        self.PATTERN_CATEGORIES = [
            "Standard Issue-Fix Workflow",
            "Contentious Debate or Revert",
            "Feature Development",
            "Exploratory Prototyping",
            "Unknown",
        ]
        self.ROLE_CATEGORIES = [
            "Core Contributor",
            "Bug Fixer",
            "Newcomer",
            "Reviewer",
            "Unknown",
        ]

        # 预先计算嵌入维度
        self.embedding_dim = self.st_model.get_sentence_embedding_dimension()

    def _encode_categorical(self, value: str, categories: List[str]) -> np.ndarray:
        """辅助函数：对单个分类值进行One-Hot编码"""
        vector = np.zeros(len(categories), dtype=int)
        try:
            # 找到值在类别列表中的索引
            index = categories.index(value)
        except ValueError:
            # 如果值不在列表中，则归为 "Unknown"
            index = categories.index("Unknown")
        vector[index] = 1
        return vector

    def _encode_roles(self, roles_list: List[Dict[str, str]]) -> np.ndarray:
        """辅助函数：对角色列表进行Multi-Hot编码"""
        vector = np.zeros(len(self.ROLE_CATEGORIES), dtype=int)
        # 如果 roles_list 为空或不存在，直接返回零向量
        if not roles_list:
            return vector

        for actor in roles_list:
            role = actor.get("inferred_role")
            try:
                index = self.ROLE_CATEGORIES.index(role)
            except (ValueError, TypeError):
                # 如果角色未知或类型错误，归为 "Unknown"
                index = self.ROLE_CATEGORIES.index("Unknown")
            vector[index] = 1  # 使用 Multi-Hot，所以只置为1
        return vector

    def encode(self, json_string: str) -> np.ndarray:
        """
        主函数：将输入的JSON字符串编码为最终的增强向量。

        Args:
            json_string (str): LLM分析结果的JSON字符串。

        Returns:
            np.ndarray: 一个扁平化的、代表整个子图的数值向量。
        """
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError:
            print("Error: Invalid JSON string provided.")
            # 返回一个零向量以避免下游任务出错
            total_dim = (
                len(self.HEALTH_CATEGORIES)
                + len(self.PATTERN_CATEGORIES)
                + len(self.ROLE_CATEGORIES)
                + 2 * self.embedding_dim
            )
            return np.zeros(total_dim)

        # 1. 编码分类特征
        health_vec = self._encode_categorical(
            data.get("subgraph_health"), self.HEALTH_CATEGORIES
        )
        pattern_vec = self._encode_categorical(
            data.get("development_pattern"), self.PATTERN_CATEGORIES
        )

        # 2. 编码角色列表
        roles_vec = self._encode_roles(data.get("key_actors_and_roles"))

        # 3. 编码文本特征 (如果文本为空，st_model会处理并返回一个向量)
        summary_text = data.get("narrative_summary", "")
        outlook_text = data.get("future_outlook", "")

        summary_embedding = self.st_model.encode(summary_text)
        outlook_embedding = self.st_model.encode(outlook_text)

        # 4. 拼接所有向量，形成最终的子图增强向量
        final_vector = np.concatenate(
            [health_vec, pattern_vec, roles_vec, summary_embedding, outlook_embedding]
        )

        return final_vector


def encode_subgraph_file(input_file: str, output_file: str, batch_size: int):
    """
    对文件中的所有子图数据进行编码，并保存为一个二维 NumPy 数组。

    Args:
        input_file (str): 输入文件路径，包含子图数据。
        output_file (str): 输出文件路径，用于保存编码后的 NumPy 数组。
        batch_size (int): 每个批次的大小，用于构建二维数组。
    """
    encoder = SubgraphEncoder()  # 初始化编码器
    encoded_batches = []  # 用于存储批次数据
    current_batch = []  # 当前批次

    # 统计文件行数，用于进度条显示
    with open(input_file, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    # 打开文件并使用 tqdm 创建进度条
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Encoding subgraphs"):
            try:
                # 解析每行数据为 JSON
                subgraph_data = json.loads(line.strip())
                analysis_str = subgraph_data.get("analysis", "{}")
                try:
                    # 尝试将 analysis 字符串解析为字典
                    analysis = json.loads(analysis_str)
                except json.JSONDecodeError:
                    # 如果解析失败，使用空字典
                    analysis = {}
                # 将 analysis 字典转换为 JSON 字符串
                analysis_json_string = json.dumps(analysis)

                # 对分析字段进行编码
                encoded_vector = encoder.encode(analysis_json_string)

                # 将编码后的向量添加到当前批次
                current_batch.append(encoded_vector)

                # 如果当前批次达到 batch_size，则保存并重置
                if len(current_batch) == batch_size:
                    encoded_batches.append(np.array(current_batch))
                    current_batch = []
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
            except Exception as e:
                print(f"Error processing line: {line}\n{e}")

    # 保存剩余的批次（如果有）
    if current_batch:
        encoded_batches.append(np.array(current_batch))

    # 将所有批次合并为一个二维数组
    encoded_array = np.vstack(encoded_batches)

    # 保存到文件
    np.save(output_file, encoded_array)
    print(f"Encoded data saved to {output_file}")


import re


def extract_parameters_from_filename(filename: str):
    """
    从文件名中提取 batch_size 和 neg_num 参数。

    Args:
        filename (str): 文件名字符串。

    Returns:
        tuple: (batch_size, neg_num)，如果提取失败则返回 (None, None)。
    """
    try:
        # 使用正则表达式提取参数
        batch_size_match = re.search(r"_bs(\d+)", filename)
        neg_num_match = re.search(r"_neg(\d+)", filename)

        # 提取并转换为整数
        batch_size = int(batch_size_match.group(1)) if batch_size_match else None
        neg_num = int(neg_num_match.group(1)) if neg_num_match else None

        return batch_size, neg_num
    except Exception as e:
        print(f"Error extracting parameters from filename: {filename}\n{e}")
        return None, None


if __name__ == "__main__":
    input_file = "tgb/DATA/thgl_software_subset/valid_neg_sample_neg1_bs600_hops5_neighbors50_llm_analysis.txt"
    batch_size, neg_num = extract_parameters_from_filename(input_file)
    batch_size = batch_size * (neg_num + 2)
    output_file = (
        input_file.split(".")[0].replace("llm_analysis", "llm_encode") + ".npy"
    )

    encode_subgraph_file(input_file, output_file, batch_size)

# if __name__ == "__main__":
#     # 你的JSON输入
#     # json_input = """
#     # {
#     #   "narrative_summary": "The user reopened a previously closed Pull Request multiple times, indicating unresolved issues or modifications needed, which reflects ongoing attempts to address complications associated with the repository.",
#     #   "development_pattern": "Contentious Debate or Revert",
#     #   "subgraph_health": "Problematic",
#     #   "key_actors_and_roles": [
#     #     {
#     #       "node_id": "2822",
#     #       "inferred_role": "Bug Fixer"
#     #     }
#     #   ],
#     #   "future_outlook": "Further adjustments and discussions are likely necessary before the Pull Request can be successfully closed or merged."
#     # }
#     # """

#     json_input = "{}"

#     # 1. 创建编码器实例
#     # (模型会在第一次创建实例时下载并加载，可能需要一些时间)
#     encoder = SubgraphEncoder()

#     # 2. 调用encode方法生成向量
#     subgraph_vector = encoder.encode(json_input)

#     # 3. 查看结果
#     print("\n--- Encoding Result ---")
#     print("Generated Vector:")
#     # print(subgraph_vector) # 打印向量数值会很长，我们主要看形状
#     print("\nVector Shape:", subgraph_vector.shape)

#     # 让我们分析一下维度的构成
#     health_dim = len(encoder.HEALTH_CATEGORIES)
#     pattern_dim = len(encoder.PATTERN_CATEGORIES)
#     role_dim = len(encoder.ROLE_CATEGORIES)
#     embedding_dim = encoder.embedding_dim  # all-MiniLM-L6-v2 是 384

#     print(f"\nVector dimensions breakdown:")
#     print(f"  Health            : {health_dim} dims")
#     print(f"  Pattern           : {pattern_dim} dims")
#     print(f"  Roles             : {role_dim} dims")
#     print(f"  Summary Embedding : {embedding_dim} dims")
#     print(f"  Outlook Embedding : {embedding_dim} dims")
#     print(f"  -------------------------------------")
#     print(
#         f"  Total Dimension   : {health_dim + pattern_dim + role_dim + 2 * embedding_dim}"
#     )
