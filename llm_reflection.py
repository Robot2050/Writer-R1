import json
from typing import List, Dict, Any
from src.llm import Agent
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm
import re
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

def call_writer_agent(lang: str, query: str, previous_response: str = None, previous_review: str = None) -> str:
    """
    调用写作 Agent：
    - 第一次：直接根据 query 写大纲
    - 后续：根据上一轮回复 + 评审意见重写
    """
    if previous_response is None or previous_review is None:
        # 首轮写作
        prompt = query
    else:
        # 之后的迭代优化：给出上一轮答案 + 评审意见，要求在此基础上重写
        if lang == "zh":
            prompt = f"""你是一名专业的中文写作助手。
在充分理解下面的原始需求与评审意见的基础上，对上一轮模型回答进行**整体重写与优化**，输出一版全新优化后的完整内容。

【具体要求】
- 不要仅做局部润色，而是根据评审意见对整体结构和内容逻辑进行系统性重构；
- 可以保留原回答中合理且高质量的要点，并以更优的方式重新组织和表达；
- 在重写内容时，要重点改进和强化评审意见中指出的不足之处；
- 直接给出重写后的最终版本内容，不要添加额外说明和解释的内容。


【原始写作需求】
{query}

【上一轮模型回答】
{previous_response}

【评审意见】
{previous_review}


仅输出重写后的最终内容：
"""
        else:
            prompt = f"""You are a professional Chinese writing assistant.
On the basis of a thorough understanding of the original writing request and the review comments below, you are to **comprehensively rewrite and refine** the previous model output, producing a fully revised and optimized version of the content.

**Specific Requirements**
* Do not limit yourself to local polishing; instead, systematically reconstruct the overall structure and logical flow of the content in accordance with the review comments;
* You may retain the reasonable and high-quality points from the previous answer, but they should be reorganized and rephrased in a superior manner;
* When rewriting, place particular emphasis on improving and strengthening the weaknesses identified in the review comments;
* Directly provide the final rewritten version of the content, without adding any additional explanations or meta-commentary.

**Original Writing Request**
{query}

**Previous Model Answer**
{previous_response}

**Review Comments**
{previous_review}

Output only the final rewritten content:
"""

    writer = Agent("You are a helpful assistant.")
    response, success = writer.run(prompt)
    if not success:
        raise RuntimeError(f"Writer agent failed!")
    return response

def _build_reviewer_prompt(lang: str, response: str, checklist: List[Dict[str, Any]]) -> str:

    checklist_str = json.dumps(checklist, ensure_ascii=False, indent=2)

    if lang == "zh":
        prompt = f"""你是一名严谨的学术评审专家，需要根据给定的 checklist，对一份模型生成的回答进行逐项评分和点评。

【任务说明】
checklist 每一项包含：
- name：指标名称；
- criteria_description：该指标的详细说明，以及各分数区间描述。

请你按照 checklist 中条目的顺序，对每一个 name 依次进行评估。对每一项，请严格按照下面格式输出：
name：<与 checklist 中 name 完全一致>
评分：<得分>/<总分>
评价和改进建议：<你的评价与改进建议>

然后继续下一项，以此类推，直到所有 checklist 条目都被评估。

【评分要求】
- 评分为整数，只输出数字，不包含其他字符；
- 评价和改进建议需要同时包含该项做得好的地方和可以改进的方向。

【输出格式要求】
- 只输出上述三行一组的重复结构，不要输出任何其他说明；
- 各项之间可以用空行分隔。


【待评估的模型回答】
{response}

【checklist】
{checklist_str}
"""
    else:
        prompt = f"""You are a rigorous academic reviewer who must evaluate a model-generated answer against a given checklist, scoring and commenting on each item individually.

【Task Description】
Each checklist item contains:
* name: the name of the evaluation criterion;
* criteria_description: a detailed description of this criterion, including the descriptions corresponding to different score ranges.

Please follow the order of the items in the checklist and evaluate each **name** in turn. For each item, you must strictly use the following output format:
name: <exactly the same as the corresponding name in the checklist>
score: <obtained_score>/<maximum_score>
comments and suggestions for improvement: <your evaluation and suggestions for improvement>

Then proceed to the next item, and so on, until all checklist entries have been evaluated.

【Scoring Requirements】
* The score must be an integer; only output the numeric value without any additional characters.
* The comments and suggestions for improvement must include both what is done well for this criterion and what can be improved.

【Output Format Requirements】
* You should output only the repeated triplets of the three lines above; do not output any additional explanations.
* You may separate different items with blank lines.

【Model Answer to Be Evaluated】
{response}

【Checklist】
{checklist_str}
"""

    return prompt

def _parse_reviewer_text(lang: str, review_text: str) -> Dict[str, Any]:
    """
    使用正则从纯文本中解析出：
        name / score / total / comment
    并计算平均得分（统一折算到 10 分制）。
    """
    text = review_text.strip()

    items = []

    if lang == "zh":
        # 只匹配：
        # 评分：得分/总分
        pattern = re.compile(
            r"(?:评分|得分|分数)\s*(?:[:：=]|为|是)?\s*"
            r"(?P<score>\d+(?:\.\d+)?)\s*\D{0,30}?\s*"
            r"(?:/|／|\\)\s*\D{0,30}?\s*"
            r"(?P<total>\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
    else:
        # 只匹配英文格式：
        # score: 得分/总分
        pattern = re.compile(
            r"(?:score|rating|grade)\s*(?:[:：=]|is|was)?\s*"
            r"(?P<score>\d+)\s*\D{0,50}?\s*"
            r"(?:/|out\s*of)\s*\D{0,50}?\s*"
            r"(?P<total>\d+)",
            re.IGNORECASE,
        )
    for m in pattern.finditer(text):
        score = m.group("score")
        total = m.group("total")

        items.append(
            {
                "score": score,
                "total": total
            }
        )

    return items

def call_reviewer_agent(lang: str, response: str, checklist: List[Dict[str, Any]]) -> Dict[str, Any]:
    prompt = _build_reviewer_prompt(lang=lang, response=response, checklist=checklist)

    reviewer = Agent("You are a helpful assistant.")
    review_text, success = reviewer.run(prompt)
    if not success:
        raise RuntimeError(f"Reviewer agent failed")

    parsed = _parse_reviewer_text(lang=lang, review_text=review_text)
    return parsed,review_text


def iterative_generation_with_checklist(
    lang: str,
    query: str,
    checklist: List[Dict[str, Any]],
    max_rounds: int = 3,
    pass_ratio: float = 0.8,
) -> Dict[str, Any]:
    """
    总流程：
    1. 用 writer_agent 生成回答（论文大纲）
    2. 用 reviewer_agent 根据 checklist 打分和点评
    3. 计算 overall_score，判断是否 >= 80%（默认 8/10）
    4. 若未达标且未超过最大轮数，则带着“上一轮回答 + 评审意见”重写
    5. 所有中间过程（每轮的 response 与 review）都保存在 history 中
    6. 返回最终结果，包括：
       - final_response: 最终论文大纲
       - final_review:   最终一轮评审结果
       - history:        所有轮次的中间过程
    """
    history: List[Dict[str, Any]] = []

    previous_response = None
    previous_review_text = None
    final_response = None
    final_review = None

    for round_idx in range(1, max_rounds + 1):
        # 休息 1 秒，避免对 API 的频繁调用
        time.sleep(1)
        
        # 1. 生成 / 重写回答
        response = call_writer_agent(
            lang=lang,
            query=query,
            previous_response=previous_response,
            previous_review=previous_review_text,
        )

        # 2. 评审打分
        review_json,overall_comment = call_reviewer_agent(
            lang=lang,
            response=response,
            checklist=checklist,
        )

        get_score=0
        total_score=0
        for item in review_json:
            try:
                score = float(item["score"])
                total = float(item["total"])
            except Exception as e:
                print(f"Error parsing score/total: {item}. Error: {e}",flush=True)
                continue

            # 合理性检查：total 必须 > 0，score 不能为负
            if total <= 0 or score < 0:
                continue
            if score > total:
                continue

            get_score += score
            total_score += total

        # 防止除零
        if total_score == 0:
            overall_score = 0.0   # 或者 raise ValueError("No valid score parsed")
        else:
            overall_score = get_score / total_score
        
        # 将这一轮的数据加入 history
        history.append(
            {
                "round": round_idx,
                "response": response,
                "review_json": review_json,
                "overall_score": overall_score,
                "overall_comment": overall_comment,
            }
        )

        # 更新“上一轮”的信息，供下一轮重写使用
        previous_response = response
        # 评审意见作为一段自然语言文本，给写作 Agent 用
        previous_review_text = overall_comment

        # 3. 判断是否达标
        if overall_score >= pass_ratio:
            final_response = response
            final_review = overall_comment
            best_score=overall_score
            break

        # 若未达标，继续下一轮；如果已经是最后一轮，直接收尾
        if round_idx == max_rounds:
            best_item = max(
                history,
                key=lambda x: x["overall_score"]
            )
            final_response = best_item["response"]
            final_review = best_item["overall_comment"]
            best_score=best_item["overall_score"]

    return {
        "final_response": final_response,
        "final_review": final_review,
        "best_score": best_score,
        "history": history,
    }

def process_one_content(content, args):
    index = content["index"]
    lang = content["lang"]
    query = content["query"]
    checklist = content["checklist"]   # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! checklist_gpt checklist
    time.sleep(2)
    try:
        result = iterative_generation_with_checklist(
            lang=lang,
            query=query,
            checklist=checklist,
            max_rounds=5,
            pass_ratio=0.9,
        )

        print(f"===== index {index} 最终评审结果 =====")
        print(json.dumps(result["final_review"], ensure_ascii=False, indent=2))

        success_record = {
            "index": index,
            "query": query,
            "checklist": checklist,
            "result": result,
            # "best_score": result["best_score"],
        }
        eval_data = {
            "index": index,
            "response": result["final_response"],
        }

        with save_lock:
            save_output([success_record], f"{args.output_folder}/success_record.jsonl")
            save_output([eval_data], f"{args.output_folder}/eval_data.jsonl")

        return {"index": index, "status": "success", "best_score": result["best_score"]}

    except Exception as e:
        error_record = content

        with save_lock:
            save_output([error_record], f"{args.output_folder}/error_record.jsonl")

        print(f"index {index} 运行过程中发生异常，已记录")
        print(e)
        return {"index": index, "status": "error", "best_score": None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process lines from an input file.")  
    parser.add_argument("--query_file", type=str, default='benchmark_query/benchmark_all.jsonl', help="Path to the query file.")
    parser.add_argument("--output_folder", type=str, default='llm_reflection/deepseek-v3.2', help="Path to the output file.")

    args = parser.parse_args()

    # === 加载文件 ===
    contents, input_cnt = load_file(args.query_file)
    print(f"共加载 {input_cnt} 条记录")

    num_workers = 5
    print(f"使用并行线程数: {num_workers}")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(process_one_content, content, args): content["index"]
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
                "status": result_info.get("status"),
                "best_score": result_info.get("best_score"),
            })

# nohup python -u llm_reflection.py >> llm_reflection/deepseek-v3.2/llm_reflection.log 2>&1 &

