# -*- coding: utf-8 -*-
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
from tqdm import tqdm
from src.llm import Agent


# -----------------------------
# Thread-safe file appending
# -----------------------------
save_lock = threading.Lock()

agent = Agent("You are a helpful assistant.")

def append_jsonl(path: str | Path, obj: Any) -> None:
    """Append a python object as one JSON line."""
    line = json.dumps(obj, ensure_ascii=False)
    with save_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def append_raw_line(path: str | Path, raw_line: str) -> None:
    """Append a raw line (already includes/doesn't include newline - we normalize)."""
    raw_line = raw_line.rstrip("\n")
    with save_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(raw_line + "\n")


# -----------------------------
# Prompt templates (tagged output for regex)
# -----------------------------
METRIC_GEN_PROMPT = """你将作为“创意文本写作任务”的评估指标生成器。
给定一个用户请求 query，请生成尽量全面、可操作且可区分的评估指标清单。每条指标必须包含：
- name：指标名称（简洁、明确、可归类，例如论文大纲生成任务中的“论文结构完整性与学术规范”、“机器学习算法创新性”等）
- reason：该指标评什么、为什么重要、如何体现（尽量给出可观测、可判定的检查点）

核心原则：
1) 指标必须充分贴合该 query 的具体主题、体裁、受众、语气风格、场景设定等；不得只给“通用写作模板式指标”。  
   - 至少生成一半的“题目特异指标”（只对该题目或非常相近题目有意义），明确评估是否抓住该题目独有的主题要点、立意角度、关键信息与隐含要求等。   
2) 通用性与特异性平衡：其余指标可覆盖语言表达、结构组织等更通用的写作质量维度，但必须与该 query 的任务目标关联起来。
3) 可评估性：所有指标必须可用于人工或自动评估，避免空泛词（如“很好/一般/有趣”）作为依据。
4) 可区分：输出 15~20 条，避免同义重复与高度重叠；每条指标应聚焦一个清晰可区分的评价点。
5) 输出格式约束：只输出规定结构的文本，不要任何解释、标题、编号、项目符号或前后缀；每条指标严格两行，顺序固定为 name 然后 reason。

严格输出格式（重复多组，不同组使用两个换行区分）：
name: <指标名>
reason: <为什么该指标重要/评什么等>


用户请求 query：
{query}
"""

SELECT_AND_DESC_PROMPT = """你将作为评估指标的筛选与融合器。
输入是一组候选评估指标（name 以及对应的 reason）和用户请求 query，你的任务是筛选并整理出最终可用的核心评估指标集合。

输入字段说明：
- name：指标名称（简洁、明确、可归类，例如论文大纲生成任务中的“论文结构完整性与学术规范”、“机器学习算法创新性”等）
- reason：该指标评什么、为什么重要、如何体现（尽量给出可观测、可判定的检查点）


工作要求：
1) 指标筛选与保留规则  
- 若某个候选指标本身已经语义完整、评估边界清晰、且对当前 query 具有明确独特价值，则应**直接保留其核心含义**（可对 name 和 reason 进行优化，但不得引入新的评估点）。  
- 若多个指标在评估对象、判定标准或关注重点上较为相近，可抽象为一个更高质量、更稳定的评估维度，则应**进行融合**，形成一个更通用但不空泛的新指标。  

2) 融合原则（当需要融合时）  
- 融合后的指标必须覆盖原有指标的关键信息，但避免简单拼接或堆砌概念。  
- 融合后的 name 应简洁、概括性强，避免冗长复合名词。  
- 融合后的 reason 应体现“融合后的统一评估逻辑”，而非逐条罗列原指标。  

3) 最终指标必须充分贴合该 query 的具体主题、体裁、受众、语气风格、场景设定等；不得只给“通用写作模板式指标”。  
   - 至少生成一半的“题目特异指标”（只对该题目或非常相近题目有意义），明确评估是否抓住该题目独有的主题要点、立意角度、关键信息与隐含要求等。   

4) 通用性与特异性平衡：其余指标可覆盖语言表达、结构组织等更通用的写作质量维度，但必须与该 query 的任务目标关联起来。

5) 可评估性：所有指标必须可用于人工或自动评估，避免空泛词（如“很好/一般/有趣”）作为依据。

6) 可区分：输出 5~8 条，避免同义重复与高度重叠；每条指标应聚焦一个清晰可区分的评价点，覆盖核心质量维度。

7) 输出格式约束：只输出规定结构的文本，不要任何解释、标题、编号、项目符号或前后缀；每条指标严格两行，顺序固定为 name 然后 reason。 

严格输出格式（重复多组，不同组使用两个换行区分）：
name: <指标名>
reason: <为什么该指标重要/评什么等>


示例（仅用于说明输出格式，不代表本题内容）：
name: 主题契合度与立意准确性
criteria_description: 评估输出是否紧扣题目主题与核心立意展开，是否准确回应题目隐含要求如价值取向、时代语境或情感基调，并通过关键情节/论点/意象等可观察内容支撑主题表达，避免跑题或泛化套话。


候选指标如下：
{metrics_block}


用户请求 query：
{query}
"""



RUBRIC_GEN_PROMPT = """你将作为评估指标的评分量表生成器。
请针对给定的评估指标 name 和对应的描述 criteria_description，构建尽可能全面、可操作、可区分的打分梯度描述，用于对模型回答用户请求的输出进行一致性评分。

输入字段说明：
- name：评估维度/指标名称，表示需要被评价的核心能力或属性。
- criteria_description：该指标的评估要点与判定依据。

输入：
- name: {name}
- criteria_description: {criteria_description}

生成要求：
1) 生成 5~8 个评分区间，覆盖从“明显不满足”到“充分满足/卓越”的完整表现谱系，尽可能做到梯度细致且区分明确。
2) 分数上限不固定，不要求以 10 分为满分；但每个区间的跨度必须为 2 分（例如 1-2、3-4、…），并保持区间连续、不重叠、严格递增。
3) 每个区间的描述应当：
   - 与 criteria_description 保持整体保持一致性；
   - 使用可观测、可验证的表述（避免空泛词，如“很好/一般”），尽量给出判定要点或典型特征；
   - 保持同一维度下各区间描述在“覆盖范围、细节深度、约束严格性”上的递进关系；
   - 边界明确：避免相邻区间描述含糊重叠，保证评审者能够稳定区分。
4) 仅输出评分区间及描述，不要输出任何解释、标题、前后缀或额外符号。

输出格式（严格遵守，仅如下结构，多行）：
1-2: ...
3-4: ...
5-6: ...
...


用户请求 query（用于理解任务语境与题目特异要求，不要在输出中复述 query）：
{query}
"""


# -----------------------------
# Regex parsers
# -----------------------------
_METRIC_RE = re.compile(
    r"""
    (?isxm)                              # i:IGNORECASE s:DOTALL x:VERBOSE m:MULTILINE
    ^\s*name\s*[:：]\s*                  # name:
    (?P<name>.+?)\s*                     # 捕获 name（非贪婪）
    (?:\r?\n)+\s*reason\s*[:：]\s*       # reason: (允许空行/缩进)
    (?P<reason>.*?)\s*                   # 捕获 reason（可多行，非贪婪）
    (?=                                  # 截断到下一个 name: 或文本结束
        (?:\r?\n)+\s*name\s*[:：]
        |
        \Z
    )
    """,
    re.IGNORECASE | re.DOTALL | re.MULTILINE | re.VERBOSE
)


RUBRIC_DESC_PATTERN = re.compile(
    r"""
    ^\s*                      # 行首及可能的空白
    \d+\s*[-–—~]\s*\d+        # 分数区间，如 1-2 / 1–2 / 1 ~ 2
    \s*[:：]\s*               # 中英文冒号
    (?P<desc>.*?)             # 只捕获描述内容（非贪婪）
    (?=                        # 前瞻：下一个区间或文本结束
        \n\s*\d+\s*[-–—~]\s*\d+\s*[:：]
        |
        \Z
    )
    """,
    re.MULTILINE | re.DOTALL | re.VERBOSE
)


@dataclass
class SelectedCriterion:
    name: str
    criteria_description: str


def parse_metrics(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in _METRIC_RE.finditer(text):
        name = m.group("name").strip()
        reason = m.group("reason").strip()
        if name:
            out.append((name, reason))
    # 去重（按 name）
    uniq: Dict[str, str] = {}
    for n, r in out:
        if n not in uniq:
            uniq[n] = r
    return [(n, uniq[n]) for n in uniq]


def parse_selected(text: str) -> List[SelectedCriterion]:
    out: List[SelectedCriterion] = []
    for m in _METRIC_RE.finditer(text):
        name = m.group("name").strip()
        desc = m.group("reason").strip()
        if name and desc:
            out.append(SelectedCriterion(name=name, criteria_description=desc))
    # 去重（按 name）
    uniq: Dict[str, SelectedCriterion] = {}
    for c in out:
        if c.name not in uniq:
            uniq[c.name] = c
    return list(uniq.values())


def parse_rubric_block(text: str) -> List[str]:
    """
    从评分量表文本中提取每个分数区间对应的描述内容（不包含分数本身）
    """
    descriptions = []
    for m in RUBRIC_DESC_PATTERN.finditer(text):
        desc = m.group("desc").strip()
        if desc.strip():
            descriptions.append(desc.strip())
    return descriptions


# -----------------------------
# LLM call wrapper (uses your Agent)
# -----------------------------
def llm_run(prompt: str) -> Tuple[str, bool]:
    response, success = agent.run(prompt)
    return response, success


# -----------------------------
# Core processing
# -----------------------------
def build_metrics_block(metrics: List[Tuple[str, str]]) -> str:
    # 给第二步用：保持清晰、可读
    lines = []
    for i, (n, r) in enumerate(metrics, 1):
        lines.append(f"{i}. name: {n}\nreason: {r}")
    return "\n\n".join(lines)



def process_one_item(
    raw_line: str,
    output_jsonl: str | Path,
    failed_jsonl: str | Path,
    inner_workers: int = 6,
) -> bool:
    """
    Returns True if success (written to output), else False (written to failed).
    """
    time.sleep(5)
    raw_line_stripped = raw_line.rstrip("\n")

    # 1) Parse input JSON
    try:
        item = json.loads(raw_line_stripped)
    except Exception:
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False

    query = item.get("query")
    if not isinstance(query, str) or not query.strip():
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False
    query = query.strip()
    print(f"\nIndex {item.get('index')}. Processing query: {query[:60]}...")

    # 2) Generate many metrics
    resp1, ok1 = llm_run(METRIC_GEN_PROMPT.format(query=query))
    if not ok1:
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False

    metrics = parse_metrics(resp1)
    if len(metrics) < 2:
        # 解析不到足够指标也算失败（避免写入低质量结构）
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False
    print(f"\nIndex {item.get('index')}. Generated {len(metrics)} candidate metrics.")

    # 3) Select & create criteria_description
    metrics_block = build_metrics_block(metrics)
    resp2, ok2 = llm_run(SELECT_AND_DESC_PROMPT.format(metrics_block=metrics_block,query=query))
    if not ok2:
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False

    selected = parse_selected(resp2)
    if not (4 <= len(selected) <= 8):
        # 允许略微偏离，但至少要 >=5
        if len(selected) < 4:
            append_raw_line(failed_jsonl, raw_line_stripped)
            return False
        # 超过10就截断（防止写太多）
        selected = selected[:8]
    print(f"\nIndex {item.get('index')}. Selected {len(selected)} final criteria.")

    # 4) Generate rubric ranges in parallel for each selected criterion
    def gen_and_fuse_one(c: SelectedCriterion) -> Optional[Dict[str, Any]]:
        # 4.1 生成尽量全面的梯度
        r3, ok3 = llm_run(
            RUBRIC_GEN_PROMPT.format(
                name=c.name,
                criteria_description=c.criteria_description,
                query=query
            )
        )
        if not ok3:
            return None

        ranges3 = parse_rubric_block(r3)
        if not ranges3 or len(ranges3) < 4:
            return None
        
        # 评分梯度以 key:value 展开（每2分一档）
        ranges_use: Dict[str, str] = {
            f"{2*i+1}-{2*i+2}": r.strip()
            for i, r in enumerate(ranges3)
        }
        # 最终结构
        out: Dict[str, Any] = {
            "name": c.name,
            "criteria_description": c.criteria_description,
        }
        # 评分梯度以 key:value 展开
        out.update(ranges_use)
        return out

    checklists: List[Dict[str, Any]] = []
    max_inner = max(1, min(inner_workers, len(selected)))
    with ThreadPoolExecutor(max_workers=max_inner) as ex:
        futs = [ex.submit(gen_and_fuse_one, c) for c in selected]
        for fut in as_completed(futs):
            res = fut.result()
            if res is not None:
                checklists.append(res)

    # 如果生成的 checklist 太少，认为失败
    if len(checklists) < 3:
        append_raw_line(failed_jsonl, raw_line_stripped)
        return False

    # 可选：按原 selected 顺序排序（更稳定）
    order = {c.name: i for i, c in enumerate(selected)}
    checklists.sort(key=lambda d: order.get(d.get("name", ""), 10**9))
    print(f"\nIndex {item.get('index')}. Generated {len(checklists)} checklists with rubrics.")
    
    # 5) Write final JSON line
    item_out = dict(item)
    item_out["checklist_gpt"] = checklists
    append_jsonl(output_jsonl, item_out)
    return True


def main(
    input_jsonl,
    output_jsonl,
    failed_jsonl,
    max_workers,
    inner_workers,
) -> None:
    """
    - Reads input jsonl lines into a list (as requested)
    - Parallel executes process_one_item with configurable concurrency
    """
    input_path = Path(input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_jsonl}")

    # # Clean output files (optional). If you want append mode, comment these out.
    # Path(output_jsonl).write_text("", encoding="utf-8")
    # Path(failed_jsonl).write_text("", encoding="utf-8")

    # Read all lines into list (per your requirement)
    with open(input_jsonl, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    # Parallel processing
    max_workers = max(1, int(max_workers))
    ok_count = 0
    fail_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                process_one_item,
                raw_line=ln,
                output_jsonl=output_jsonl,
                failed_jsonl=failed_jsonl,
                inner_workers=inner_workers,
            )
            for ln in lines
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="processing", unit="item"):
            if fut.result():
                ok_count += 1
            else:
                fail_count += 1

    print(f"Done. success={ok_count}, failed={fail_count}")
    print(f"Output: {output_jsonl}")
    print(f"Failed: {failed_jsonl}")


if __name__ == "__main__":
    # 你可以按需调整并行度
    # 外层：每条 query 并行数 max_workers
    # 内层：每条 query 里对多个指标并行生成量表 inner_workers
    # 记录运行时间
    start_time = time.time()
    print('The Model is longcat-flash-chat')

    main(
        input_jsonl="/users/u2023000896/WritingBench-main/HelloBench-main/generate_checklist/longcat-flash-chat/all_filtered_from_6doc_checklist_failed.jsonl",
        output_jsonl="/users/u2023000896/WritingBench-main/HelloBench-main/generate_checklist/longcat-flash-chat/fail_add/all_filtered_checklist.jsonl",
        failed_jsonl="/users/u2023000896/WritingBench-main/HelloBench-main/generate_checklist/longcat-flash-chat/fail_add/all_filtered_checklist_failed.jsonl",
        max_workers=5,
        inner_workers=5,
    )
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

# nohup python -u checklist_llm.py >> checklist_llm.log 2>&1 &
