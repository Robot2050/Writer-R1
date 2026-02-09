import json
import os
import argparse
from tqdm import tqdm
from openai import OpenAI
import time
import requests

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
    if os.path.isfile(file_name):
        with open(file_name, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            return records, len(records)
    return [], 0

# llm_model='vllm'  # vllm  huawei
# if llm_model=='vllm':
#     from vllm import LLM, SamplingParams
#     from transformers import AutoModelForCausalLM, AutoTokenizer
#     model_name_or_path="/users/u2023000898/model/Qwen3-8B" 
#     tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#     sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05) 
#     model = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 

# === Your code here ===

def writer(query):
    time.sleep(2)  # 避免请求过快
    try:
        # base_url = "https://api.modelarts-maas.com/v1" # API地址
        # api_key = "" # 把yourApiKey替换成已获取的API Key

        # client = OpenAI(api_key=api_key, base_url=base_url)

        # response = client.chat.completions.create(
        #     model = "qwen3-32b", # DeepSeek-V3,  deepseek-v3.2-exp, deepseek-v3.1, Kimi-K2, longcat-flash-chat
        #     messages = [
        #         {"role": "system", "content": "You are a helpful assistant"},
        #         {"role": "user", "content": query}, 
        #     ],
        #     temperature=0.7,
        #     top_p=0.8,
        #     presence_penalty=1.05,
        # )
        # return response.choices[0].message.content.strip()
        
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": query}
        # ]
        # text = tokenizer.apply_chat_template(
        #     messages,
        #     tokenize=False,
        #     add_generation_prompt=True,
        #     enable_thinking=False,  # Set to False to strictly disable thinking
        # )
        # outputs = model.generate([text], sampling_params)
        # for output in outputs:
        #     generated_text = output.outputs[0].text
        # response = generated_text
        # return response
        
        url = "https://api.modelarts-maas.com/v2/chat/completions"  # API地址
        api_key = ""  # 把MAAS_API_KEY替换成已获取的API Key
        # Send request.
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        data = {
            "model": "deepseek-v3.2",  # model参数
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.7,
            "top_p": 0.8,
            "thinking": {
                    "type": "enabled"  # 是否开启深度思考模式，默认开启
                }
        }
        response = requests.post(url, headers=headers, data=json.dumps(data))
        return json.loads(response.text)["choices"][0]["message"]["content"].strip()
    
    except Exception as e:
        print(f"error: {e}")
        
        


def process(id_query_map, out_file):
    records, existing_count = load_file(out_file)
    cnt = existing_count
    contents, input_cnt = load_file(id_query_map)
    with tqdm(total=input_cnt, initial=0, desc=f"Processing {id_query_map.split('/')[-1]}") as pbar:
        for i, content in enumerate(contents):
            try:
                if existing_count > 0 and i < existing_count: 
                    pbar.update()
                    continue
                data = {"index": content["index"]}
                query = content["query"]
                data["response"] = writer(query)
                save_output([data], out_file)
                cnt += 1
                pbar.update()
            except Exception as e:
                print(f"Error processing index {content['index']}: {e}")
                pbar.update()
                continue

    print(f"CNT: {cnt}")
    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process lines from an input file.")
    parser.add_argument("--query_file", type=str, help="Path to the query file.")
    parser.add_argument("--output_file", type=str, help="Path to the output file.")

    args = parser.parse_args()

    process(args.query_file, args.output_file)

# nohup python generate_response.py --query_file benchmark_query/benchmark_all.jsonl --output_file responses/deepseek-v3.2.jsonl >> responses/deepseek-v3.2.log 2>&1 &






