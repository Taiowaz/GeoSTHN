import os
import pickle
import json
from openai import OpenAI
from src.LLM.prompt import create_software_dev_prompt
from tqdm import tqdm  # 引入 tqdm 库用于显示进度条

os.chdir("/root/LLM-CDHG")

# 获取 API 密钥
api_key = os.environ.get("OPENAI_API_KEY")

# 初始化同步 OpenAI 客户端
client = OpenAI(api_key=api_key)

nodetypes = dict(json.load(open("tgb/DATA/thgl_software_subset/node_types.json")))
edgetypes = dict(json.load(open("tgb/DATA/thgl_software_subset/edge_types.json")))


def get_llm_analysis(prompt: str, model: str = "gpt-4o-mini"):
    """
    同步调用 LLM API 获取分析结果
    """
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def process_and_append(data_dict, output_file, log_file, subgraph_index, dict_index):
    """
    处理单个字典并将结果追加到文本文件，同时记录进度到日志文件
    """
    prompt = create_software_dev_prompt(data_dict, nodetypes, edgetypes)
    if prompt is None:
        analysis_result = "{}"
    else:
        try:
            raw_result = get_llm_analysis(prompt)
            # 去除换行符和可能存在的 ```json 或 ```
            cleaned_result = (
                raw_result.replace("\n", "").replace("```json", "").replace("```", "")
            )
            analysis_result = cleaned_result
        except json.JSONDecodeError:
            # 标记解析失败
            analysis_result = (
                f"解析失败: 返回结果不是有效的 JSON\n原始结果: {raw_result}"
            )

    # 构建一个大的 JSON 字符串
    result_entry = {
        "position": (subgraph_index, dict_index),
        "analysis": analysis_result,
    }

    # 将结果追加到文本文件
    with open(output_file, "a") as f:
        f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

    # 记录当前进度到日志文件
    with open(log_file, "w") as log:
        log.write(f"{subgraph_index},{dict_index}\n")


def main():
    # 加载数据文件
    fn = "/root/LLM-CDHG/tgb/DATA/thgl_software_subset/valid_neg_sample_neg1_bs600_hops5_neighbors50.pickle"
    # subgraphs = pickle.load(open(fn, "rb"))[0][0][194]
    # subgraphs = [[subgraphs]]

    subgraphs = pickle.load(open(fn, "rb"))[0]

    # 输出文件路径
    output_file = fn.split(".")[0] + "_llm_analysis.txt"
    log_file = "llm_enhence_progress_log.txt"

    # 检查文件是否存在
    if not os.path.exists(output_file):
        # 如果文件不存在，创建一个空文件
        open(output_file, "w").close()

    # 检查日志文件是否存在
    if os.path.exists(log_file):
        # 如果日志文件存在，读取最后的进度
        with open(log_file, "r") as log:
            last_progress = log.read().strip()
            if last_progress:
                last_subgraph_index, last_dict_index = map(
                    int, last_progress.split(",")
                )
                last_dict_index += 1
                if last_dict_index >= len(subgraphs[last_subgraph_index]):
                    last_subgraph_index += 1
                    last_dict_index = 0
            else:
                last_subgraph_index, last_dict_index = 0, 0
    else:
        last_subgraph_index, last_dict_index = 0, 0

    # 使用 tqdm 显示总进度
    for i, subgraph in enumerate(
        tqdm(subgraphs, desc="总进度", initial=last_subgraph_index)
    ):
        if i < last_subgraph_index:
            continue  # 跳过已处理的子列表

        for j, data_dict in enumerate(
            tqdm(subgraph, desc=f"子列表 {i + 1} 进度", initial=last_dict_index)
        ):
            if i == last_subgraph_index and j < last_dict_index:
                continue  # 跳过已处理的字典

            # 处理并保存结果，同时记录进度
            process_and_append(data_dict, output_file, log_file, i, j)

    print(f"所有分析结果已保存到 {output_file} 文件中。")


# 运行主函数
if __name__ == "__main__":
    main()
