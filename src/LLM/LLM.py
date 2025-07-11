import torch
import os
from openai import OpenAI
from typing import List, Dict, Union, Tuple

os.environ["http_proxy"] = "http://10.61.2.90:1082"
os.environ["https_proxy"] = "http://10.61.2.90:1082"


class RelationEmbeddingGenerator(torch.nn.Module):
    """
    一个用于调用LLM API，为图中的关系（边类型）生成语义嵌入的工具类。
    此版本优化了prompt的格式，使其保持统一。
    """

    def __init__(self, hid_dim: int, dataset_name: str, device: torch.device):
        super().__init__()
        self.hid_dim = hid_dim
        self.dataset_name = dataset_name
        self.device = device

        try:
            self.client = OpenAI()
        except Exception as e:
            print(f"错误：无法初始化OpenAI客户端。请确保您的API密钥已正确配置。")
            print(f"具体错误: {e}")
            self.client = None

        self.emb_model_name = "text-embedding-3-small"

    def generate_embeddings(self) -> List[torch.Tensor]:
        if not self.client:
            raise ValueError("OpenAI client 未初始化，无法继续。")

        # 使用一个元组列表来结构化地存储“关系名”和“说明”
        relation_definitions: List[Tuple[str, str]] = []

        # --- 为不同数据集定义各自的关系类型和描述 ---
        if "thgl-github" in self.dataset_name or "thgl-software" in self.dataset_name:
            print(f"检测到 {self.dataset_name} 数据集，正在准备14种关系类型的定义...")
            relation_definitions = [("", "")] * 14

            # 严格按照您提供的ID顺序填充
            relation_definitions[0] = (
                "User closes Issue",
                "In a software development graph, this relation represents a resolving action, implying the problem is solved, invalid, or a duplicate.",
            )
            relation_definitions[1] = (
                "Issue closed in Repository",
                "This relation indicates the event that an issue has been closed within a specific repository, marking the completion of a task in the project's context.",
            )
            relation_definitions[2] = (
                "Issue belongs to Repository",
                "In a software development graph, this relation defines the hierarchical ownership where an issue is contained within a specific project repository.",
            )
            relation_definitions[3] = (
                "User opens Pull Request",
                "This relation is a core collaborative action, where a user proposes code changes to a repository for review.",
            )
            relation_definitions[4] = (
                "Pull Request belongs to Repository",
                "This relation defines that a pull request is a proposal to merge code into a specific target repository.",
            )
            relation_definitions[5] = (
                "User opens Issue",
                "This relation signifies the creation of a new task, bug report, or feature request by a user, initiating a workflow.",
            )
            relation_definitions[6] = (
                "User closes Pull Request",
                "This relation, without merging, indicates the cancellation or withdrawal of proposed code changes.",
            )
            relation_definitions[7] = (
                "Pull Request closed in Repository",
                "This event marks the closing of a pull request within its repository, either by merging or rejection.",
            )
            relation_definitions[8] = (
                "User reopens Pull Request",
                "This relation signifies the reactivation of a previously closed pull request for further discussion or changes.",
            )
            relation_definitions[9] = (
                "Pull Request reopened in Repository",
                "This event shows a pull request has been reopened in its repository, bringing proposed changes back into consideration.",
            )
            relation_definitions[10] = (
                "User reopens Issue",
                "This relation indicates that a previously closed issue is being reactivated, suggesting the initial resolution was insufficient.",
            )
            relation_definitions[11] = (
                "Issue reopened in Repository",
                "This event shows that an issue has been reopened within its repository, signaling a task's return to an active state.",
            )
            relation_definitions[12] = (
                "User added to Repository",
                "This relation represents a user gaining contributor or member access to a project, a significant event for collaboration.",
            )
            relation_definitions[13] = (
                "Repository forks Repository",
                "This relation signifies a user creating a personal copy of another repository to freely experiment with changes without affecting the original project.",
            )

        elif "thgl-forum" in self.dataset_name:
            print(f"检测到 {self.dataset_name} 数据集，正在准备2种关系类型的定义...")
            relation_definitions = [("", "")] * 2
            relation_definitions[0] = (
                "user replies to user",
                "In a social forum graph, this relation represents a direct, conversational interaction, indicating engagement and discussion between two individuals.",
            )
            relation_definitions[1] = (
                "user posts to subreddit",
                "In a social forum graph, this relation represents content creation and contribution to a specific community topic.",
            )

        elif "thgl-myket" in self.dataset_name:
            print(f"检测到 {self.dataset_name} 数据集，正在准备2种关系类型的定义...")
            relation_definitions = [("", "")] * 2
            relation_definitions[0] = (
                "user installs an app",
                "In an app market graph, this relation is a strong positive signal indicating a user's choice and intent to use the application.",
            )
            relation_definitions[1] = (
                "user updates an app",
                "In an app market graph, this relation signifies user retention and continued engagement with an application they have previously installed.",
            )

        else:
            print(f"警告: 数据集 '{self.dataset_name}' 没有预设的定义。将返回空列表。")
            return []

        print(
            f"正在为 {len(relation_definitions)} 种关系类型生成嵌入向量 (模型: {self.emb_model_name})..."
        )

        relation_embeddings_list: List[torch.Tensor] = []
        for i, (rel_name, rel_desc) in enumerate(relation_definitions):
            print(f"  - 正在处理关系 ID: {i} ('{rel_name}')")
            if not rel_name or not rel_desc:
                print(f"    - 警告: ID {i} 的定义不完整。将使用零向量作为占位符。")
                relation_embeddings_list.append(torch.zeros(self.hid_dim))
                continue

            # 【优化】使用统一的模板格式化最终的prompt
            final_prompt = f"{rel_name}: {rel_desc}"

            try:
                emb = (
                    self.client.embeddings.create(
                        model=self.emb_model_name,
                        input=final_prompt,
                        dimensions=self.hid_dim,
                    )
                    .data[0]
                    .embedding
                )
                emb_tensor = torch.tensor(emb, dtype=torch.float32).to(self.device)
                relation_embeddings_list.append(emb_tensor)
                print(f"    - ID {i} 的嵌入生成成功。")
            except Exception as e:
                print(f"    - 错误: 调用API为 ID {i} 生成嵌入时失败。错误信息: {e}")
                print(f"    - 将为 ID {i} 使用零向量作为占位符。")
                relation_embeddings_list.append(
                    torch.zeros(self.hid_dim, device=self.device)
                )

        return relation_embeddings_list


def main():
    """
    主执行函数，遍历所有目标数据集，为它们分别生成并保存关系嵌入文件。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    embedding_dim = 1536

    target_datasets = [
        "thgl-github-subset",
        "thgl-software-subset",
        "thgl-forum-subset",
        "thgl-myket-subset",
    ]

    for dataset_name in target_datasets:
        print("\n" + "#" * 70)
        print(f"### 开始处理数据集: {dataset_name} ###")
        print("#" * 70)

        generator = RelationEmbeddingGenerator(embedding_dim, dataset_name, device)
        relation_embeddings_list = generator.generate_embeddings()

        if not relation_embeddings_list:
            print(f"未能为 {dataset_name} 生成任何嵌入，跳过保存步骤。")
            continue

        save_dir = os.path.join("tgb/DATA", dataset_name.replace("-", "_"))
        os.makedirs(save_dir, exist_ok=True)

        save_path_relation = os.path.join(
            save_dir,
            f"relation_embs_emb{embedding_dim}.pt",
        )

        torch.save(relation_embeddings_list, save_path_relation)
        print("-" * 70)
        print(f"✅ 第一步成功完成于数据集: {dataset_name}!")
        print(f"   关系嵌入列表已保存至: {save_path_relation}")
        print(f"   共保存了 {len(relation_embeddings_list)} 个关系的嵌入。")
        print("-" * 70)


if __name__ == "__main__":
    main()
