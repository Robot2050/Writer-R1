import json
from src.llm import Agent
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import time


save_lock = threading.Lock() # 用于保护 save_output 的并发写文件

def save_output(output, file_name):
    """
    Saves output data to a specified file in JSONL format.
    """
    with open(file_name, 'a', encoding='utf-8') as f:
        for record in output:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def load_file(file_name):
    """
    Loads JSONL lines from a file into a list of dictionaries.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f]
        return records, len(records)


def process_one_content(content, args):
    index = content["id"]
    query = content["instruction"]
    
    time.sleep(10)
    try:
        if len(query.split()) > 8000: 
            raise RuntimeError(f"index {index} query len {len(query)} exceeds the limit")    
        prompt = f"""你是一名“Query Rewriter”。你的任务是：根据输入的原始 query，生成一条**语义相关但内容明显不同**的新 query，用于构造无训练集场景下的测试集改写版本。

**目标**
* 新 query 必须与原 query **总字符串长度一致**，任务类型/意图相关。
* 但新 query 的**表述、材料、设定、细节**等必须发生**大幅变化**，保证**不一样**。


### 你必须遵守的改写规则
1. **只相关，不相似**
   * 禁止复制原句、原段、原结构的“轻微改写”。
   * 需要在“保持意图一致”的前提下进行**重构**：换说法、换组织方式、换材料细节、换示例内容等。
   * 目标：让人一眼看出“这是另一个需求/另一份材料”，但仍属于同一任务范畴。

2. **原 query 里如果包含“材料/梗概/片段/数据/背景文本”等**
   * 必须生成一套**全新的材料**，而不是在原材料上改几个词。
   * 新材料与原材料应当**主题相关但设定、人物、地点、事件链**显著不同。

3. **保留高层意图与难度阶**
   * 原 query 与新 query 仍然是同一类型产出。
   * 字数、结构、风格等约束可以调整，但要保持同等难度。

4. **语言一致性**
   * 原 query 是中文 → 输出中文；英文 → 输出英文为主；中英混合 → 保持混合比例大致一致。
   * 不要把中文任务完全改成英文任务，反之亦然。

5. **输出必须可直接用作新 query**
   * 新 query 必须像原 query 是一个清晰的写作指令。
   * 直接给出重写后的最终版本内容，不要输出任何额外说明和解释的内容。


【原始 query】
{query}


在生成最终输出前，重新回顾最开始的重写规则。直接给出重写后的最终版本内容，新 query 必须与原 query **总字符串长度一致**，不要输出任何额外说明和解释的内容。
"""
        
        writer = Agent("You are a helpful assistant.")
        rewrite_query, success = writer.run(prompt)
        if not success:
            raise RuntimeError(f"Writer agent failed!")
        
        success_record = {
            "id": index,
            "len_original_query": len(query),
            "len_rewrite_query": len(rewrite_query),
            "instruction": rewrite_query,
            "original_query": query
        }

        with save_lock:
            save_output([success_record], f"{args.output_folder}/success_record.jsonl")

        return {"index": index, "status": "success"}

    except Exception as e:
        error_record = content

        with save_lock:
            save_output([error_record], f"{args.output_folder}/error_record.jsonl")

        print(f"index {index} 运行过程中发生异常，已记录")
        print(e)
        return {"index": index, "status": "error"}



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")   
    parser.add_argument("--query_file", type=str, default='/users/u2023000896/WritingBench-main/HelloBench-main/generate_data/benchmark.jsonl', help="Path to the query file.")
    parser.add_argument("--output_folder", type=str, default='data/output/deepseek-v3.2-think-10', help="Path to the output folder.")

    args = parser.parse_args()

    # === 加载文件 ===
    contents, input_cnt = load_file(args.query_file)
    print(f"共加载 {input_cnt} 条记录")

    num_workers = 3
    print(f"使用并行线程数: {num_workers}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(process_one_content, content, args): content["id"]
            for content in contents
        }
        for future in tqdm(
            as_completed(future_to_index),
            total=len(future_to_index),
            desc="Processing",
            unit="task"
        ):
            # 这里不再用 try/except，直接拿结果
            result_info = future.result()  # process_one_content 内部已处理异常并返回 status

            # 只输出 index 和 status
            print({
                "index": result_info.get("index"),
                "status": result_info.get("status")
            })
            
# nohup python -u generate_data.py >> data/output/deepseek-v3.2-think-10.log 2>&1 &

# 增加重试重写机制，统一使用筛选机制


