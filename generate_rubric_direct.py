import json
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path="" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05,max_tokens=16384) 
model = LLM(model=model_name_or_path,tensor_parallel_size=2, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 

checklist_prompt = """你将作为“模型回答质量评估清单生成器”。
你的任务是：根据用户给定的 query，生成一组专业、严谨、可操作、可区分的评分指标 checklist，用于评估模型针对该 query 所生成回答的质量。
你需要深入理解 query 的任务类型、专业领域、隐含目标、关键约束、预期输出形式、可能的高质量回答特征，以及常见低质量回答问题等。然后据此构建一套能够稳定评价模型回答优劣的评分清单。


## 仅输出评分清单，不要输出其他任何解释或额外符号。输出格式必须严格如下：
[
    {{
        "name": "评估维度名称",
        "criteria_description": "该维度的具体评价说明。",
        "1-2": "最低表现描述。",
        "3-4": "较低表现描述。",
        "5-6": "基本达标表现描述。",
        "7-8": "良好表现描述。",
        "9-10": "优秀表现描述。",
        ......
    }},
    ......
]



## 核心生成原则

1. 评估维度设计
   - 指标必须充分贴合该 query 的具体主题、体裁、受众、语气风格、场景设定等；不得只给“通用写作模板式指标”。  
   - 至少生成一半的“题目特异指标”（只对该题目或非常相近题目有意义），明确评估是否抓住该题目独有的主题要点、立意角度、关键信息与隐含要求等。   
        - 通用性与特异性平衡：其余指标可覆盖语言表达、结构组织等更通用的写作质量维度，但必须与该 query 的任务目标关联起来。
   - 评估维度应覆盖高质量回答的关键组成部分。
   - 不同维度之间应尽量互补、低重叠，避免多个指标反复评价同一问题。
   - 对复杂、专业或研究型 query，应生成更细粒度、更领域化的维度。
   - 对简单 query，应保持指标数量适中，但仍需覆盖核心质量要求。
   - 通常生成 5~8 个评估维度，不要为了凑数量生成重复、空洞或边界不清的指标。

2. criteria_description 要求
   - 每个 criteria_description 必须准确对应 name。
   - 描述应说明该指标用于评估模型回答是否满足哪些具体要求。
   - 必须结合 query 的题目特异性，而不是泛泛描述。

3. 评分区间生成要求
   - 每个指标必须生成 5~8 个评分区间，覆盖从“明显不满足”到“充分满足/卓越”的完整表现谱系，尽可能做到梯度细致且区分明确。
   - 分数上限不固定，不要求以 10 分为满分；但每个区间的跨度必须为 2 分（例如 1-2、3-4、…），并保持区间连续、不重叠、严格递增。
   - 每个区间的描述应当：
        - 与 criteria_description 保持整体保持一致性；
        - 使用可观测、可验证的表述（避免空泛词，如“很好/一般”），尽量给出判定要点或典型特征；
        - 保持同一维度下各区间描述在“覆盖范围、细节深度、约束严格性”上的递进关系；
        - 边界明确：避免相邻区间描述含糊重叠，保证评审者能够稳定区分。


## 用户请求 query
{query}



仔细分析，输出评分清单：
"""


def checklist(query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    outputs = model.generate([text], sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
    response = generated_text.strip()
    print(f"Generated checklist: {[response]}", flush=True)
    return response


def process_jsonl(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line_num, line in enumerate(fin, start=1):
            print(f"Processing line {line_num}...", flush=True)
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"第 {line_num} 行不是合法 JSON: {e}")

            query = item.get("query")

            # 用 checklist(query) 的结果覆盖 checklist 字段
            item["checklist"] = checklist(checklist_prompt.format(query=query))

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    input_file = "/mnt/afs/chending_zjh/LlamaFactory/writing/benchmark_query/benchmark_all.jsonl"
    output_file = "/mnt/afs/chending_zjh/LlamaFactory/writing/benchmark_query/benchmark_all_Qwen-1.7B.jsonl"

    process_jsonl(input_file, output_file)



# CUDA_VISIBLE_DEVICES=4,5 nohup python generate_checklist1.py >> benchmark_query/benchmark_all_Qwen-1.7B.log 2>&1 &

