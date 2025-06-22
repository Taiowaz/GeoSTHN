import datetime

NODE_TYPE_DESCRIPTIONS = {
    "user": "A GitHub user, representing an individual who interacts with the repository.",
    "repo": "A GitHub repository, representing a project or codebase.",
    "issue": "An issue within a repository, typically representing a problem or a feature request.",
    "pr": "A pull request, representing a proposed change to the codebase.",
}

EDGE_TYPE_DESCRIPTIONS = {
    "U_SE_C_I": "A User (ID: {src_node_id})'s action results in an Issue (ID: {dst_node_id}) being in a 'Closed' state.",
    "I_AO_C_R": "A 'Closed' Issue (ID: {src_node_id}) is associated with or belongs to a Repository (ID: {dst_node_id}).",
    "I_AO_O_R": "An 'Open' Issue (ID: {src_node_id}) is associated with or belongs to a Repository (ID: {dst_node_id}).",
    "U_SO_O_P": "A User (ID: {src_node_id}) submits or opens a Pull Request (ID: {dst_node_id}) that is in an 'Open' state.",
    "P_AO_O_R": "An 'Open' Pull Request (ID: {src_node_id}) is associated with or targets a Repository (ID: {dst_node_id}).",
    "U_SE_O_I": "A User (ID: {src_node_id}) creates or opens an Issue (ID: {dst_node_id}) that is in an 'Open' state.",
    "U_SO_C_P": "A User (ID: {src_node_id})'s action results in a Pull Request (ID: {dst_node_id}) being in a 'Closed' state.",
    "P_AO_C_R": "A 'Closed' Pull Request (ID: {src_node_id}) is associated with or targets a Repository (ID: {dst_node_id}).",
    "U_SO_R_P": "A User (ID: {src_node_id}) reopens a previously closed Pull Request (ID: {dst_node_id}).",
    "P_AO_R_R": "A 'Reopened' Pull Request (ID: {src_node_id}) is associated with or targets a Repository (ID: {dst_node_id}).",
    "U_SE_RO_I": "A User (ID: {src_node_id}) reopens a previously closed Issue (ID: {dst_node_id}).",
    "I_AO_RO_R": "A 'Reopened' Issue (ID: {src_node_id}) is associated with or belongs to a Repository (ID: {dst_node_id}).",
    "U_CO_A_R": "A User (ID: {src_node_id}) is added as a 'Collaborator' or member to a Repository (ID: {dst_node_id}).",
    "R_FO_R": "A Repository (ID: {src_node_id}) is a 'Fork Of' another Repository (ID: {dst_node_id}).",
}

PROMPT_TEMPLATE = """
# [1. ROLE & TASK DEFINITION]
You are an expert graph analyst specializing in software development ecosystems. Your task is to analyze a temporal, heterogeneous interaction subgraph from the `thgl-software` ecosystem. Your primary goal is to extract high-level semantic and narrative features that are difficult for Graph Neural Networks (GNNs) to capture from structure alone. The output will be used to enrich GNN node representations for downstream tasks like link prediction.

# [2. BACKGROUND CONTEXT & DOMAIN KNOWLEDGE]
You are analyzing data that models the software development lifecycle on platforms like GitHub. Here are some key concepts to guide your analysis:
- **Healthy Workflow:** A common positive pattern is an 'Issue' being opened, followed by a 'Pull Request' (PR) being opened to address it. The PR is then closed (ideally merged), which in turn leads to the original Issue being closed.
- **Problematic Interactions:** Frequent reopening of Issues or PRs can indicate conflicts, bugs in the proposed code, or unclear requirements.
- **User Roles & Intent:** Pay attention to the roles users play. Some users primarily report issues, while others actively fix them. A user fixing an issue they did not open is a strong positive signal of collaboration. Infer the primary intent or role of key actors within this subgraph.
- **Repository Dynamics:** A 'Fork' indicates a user creating a personal copy of a repository, often to work on a PR. A user being added as a 'Collaborator' is a significant event, granting them write access.

# [3. SUBGRAPH DATA]
Below is the translated information for a specific subgraph. The timeline is relative to the central node's main event time (T=0).

---
{translated_subgraph_text}
---

# [4. INSTRUCTIONS & OUTPUT FORMAT]
Analyze the subgraph's interactions, timeline, and user roles. Your goal is to distill the complex sequence of events into a structured summary. This summary should capture the semantic essence of the interactions—the "story" of the subgraph. Focus on the narrative, user intent, and overall process health.
Please output the json object strictly, DO NOT add any text before or after the json object.
You MUST provide your analysis strictly in the following JSON format. Do not add any text before or after the JSON object.

{{
  "narrative_summary": "In one or two sentences, describe the story or the main process depicted in this subgraph. For example: 'A user reported an issue, another developer forked the repo to submit a fix via a pull request, which successfully resolved the initial issue.'",
  "development_pattern": "Categorize the dominant development pattern observed. Use one of the following: 'Standard Issue-Fix Workflow', 'Collaborative Feature Development', 'Bug Triage and Discussion', 'Contentious Debate or Revert', 'Stalled or Abandoned Work', 'Proactive Refactoring', 'Initial Setup or Onboarding'.",
  "subgraph_health": "Assess the overall health and efficiency of the interactions. Use one of the following: 'Healthy', 'Efficient', 'Concerning', 'Problematic', 'Neutral'.",
  "key_actors_and_roles": [
    {{
      "node_id": "The ID of a key user or repository in the subgraph.",
      "inferred_role": "Describe the primary role this actor played *in this specific subgraph*. For example: 'Issue Reporter', 'Core Contributor', 'Bug Fixer', 'Reviewer', 'New Contributor'."
    }}
  ],
  "future_outlook": "Based on the subgraph's current state, provide a general outlook on what might happen next. For example: 'The PR is likely to be reviewed and merged soon', 'Further discussion is needed to clarify the issue requirements', or 'The work seems abandoned and is unlikely to progress without new input.'"
}}
"""


def translate_subgraph_to_text(
    subgraph: dict, node_types: dict, edge_types: dict
) -> str:
    # 创建一个从子图本地索引到全局节点ID的映射
    local_to_global_id = {i: node_id for i, node_id in enumerate(subgraph["nodes"])}

    # 描述根节点
    root_id = subgraph["root_node"]
    root_type = node_types.get(str(root_id), "unknown_type")
    root_time_readable = datetime.datetime.fromtimestamp(
        subgraph["root_time"]
    ).strftime("%Y-%m-%d %H:%M:%S")

    description = f"Central Node: {root_type} (ID: {root_id})\n\n"
    description += "Interaction Timeline (most recent first):\n"

    # 遍历所有边来描述交互
    for i in range(subgraph["num_edges"]):
        edge_id = subgraph["eid"][i]

        # 查找边的类型代码，再通过代码查找完整的描述
        edge_type_code = edge_types.get(str(edge_id), "UNKNOWN_CODE")
        edge_description = EDGE_TYPE_DESCRIPTIONS.get(
            edge_type_code, "An unknown interaction occurred."
        )

        # 获取时间差
        # 假设时间与目标节点相关，这是时序egocentric图中常见的处理方式
        dst_local_idx = subgraph["col"][i]
        src_local_idx = subgraph["row"][i]
        src_node_id = local_to_global_id[src_local_idx]
        src_node_type = node_types.get(str(src_node_id), "unknown_type")
        dst_node_id = local_to_global_id[dst_local_idx]
        dst_node_type = node_types.get(str(dst_node_id), "unknown_type")
        time_delta_seconds = subgraph["dts"][dst_local_idx]

        # 填充占位符
        edge_description = edge_description.format(
            src_node_id=src_node_id, dst_node_id=dst_node_id
        )

        # 修改描述格式，去除边类型、源节点和目标节点信息
        description += f"- At T-{int(time_delta_seconds)}s: {edge_description}\n"

    return description


def create_software_dev_prompt(
    subgraph: dict, node_types: dict, edge_types: dict
) -> str:
    """
    根据输入的子图、节点类型和边类型，生成一个完整的提示词。

    Args:
        subgraph (dict): 包含子图结构和时序信息的字典。
        node_types (dict): 映射 {节点ID: 节点类型字符串} 的字典，例如 {101: 'user'}.
        edge_types (dict): 映射 {边ID: 边类型代码字符串} 的字典，例如 {171: 'U_SE_C_I'}.

    Returns:
        str: 一个完整的、可以直接发送给LLM的提示词字符串。
    """
    if subgraph["num_edges"] == 0:
        return None
    # 1. 调用辅助函数，将子图数据翻译成描述性文本
    translated_text = translate_subgraph_to_text(subgraph, node_types, edge_types)

    # 2. 将翻译好的文本嵌入到主模板中
    final_prompt = PROMPT_TEMPLATE.format(translated_subgraph_text=translated_text)

    return final_prompt


# ------------------- 示例用法 -------------------
if __name__ == "__main__":
    # 1. 准备模拟的输入数据
    sample_subgraph = {
        "row": [0, 0, 0, 0],
        "col": [1, 2, 3, 4],
        "eid": [171, 116, 88, 85],
        "nodes": [25, 212, 149, 109, 105],
        "dts": [0.0, 256.0, 768.0, 1024.0, 1152.0],
        "num_nodes": 5,
        "num_edges": 4,
        "root_node": 25,
        "root_time": 1704087000.0,
    }

    sample_node_types = {
        "25": "user",
        "212": "repo",
        "149": "issue",
        "109": "pr",
        "105": "user",
    }

    sample_edge_types = {
        "171": "U_SE_C_I",
        "116": "I_AO_C_R",
        "88": "U_SO_O_P",
        "85": "P_AO_O_R",
    }

    # 2. 调用主函数生成提示词
    generated_prompt = create_software_dev_prompt(
        sample_subgraph, sample_node_types, sample_edge_types
    )

    # 3. 打印结果进行验证
    print("------------------- 生成的最终提示词 -------------------")
    print(generated_prompt)
