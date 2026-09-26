import requests
import time
from typing import Callable
from openai import OpenAI


class Agent(object):
    def __init__(self,system_prompt: str = None):
        self.system_prompt = system_prompt
    
    # 需要自己编辑这个文件，选择合适的大模型调用，返回字符串结果
    def call_api(self, query: str,top_p: float,temperature: float,max_length: int) -> str:  
        base_url = "" # API
        api_key = "" # your Api Key
        client = OpenAI(api_key=api_key, base_url=base_url)

        response = client.chat.completions.create(
            model = "qwen3-32b", 
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}, 
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length,
            presence_penalty=1.05,
        )
        return response.choices[0].message.content.strip()

    def call_llm(self,
             messages: str,
             top_p: float = 0.8,
             temperature: float = 0.7,
             max_length: int = 24576):
        attempt = 0
        max_attempts = 5
        wait_time = 3

        while attempt < max_attempts:
            try:
                response = self.call_api(
                    query=messages,
                    top_p=top_p,
                    temperature=temperature,
                    max_length=max_length
                )
                return response
            except Exception as e:
                print(f"Attempt {attempt+1}: Request failed due to network error: {e}, retrying...")

            time.sleep(wait_time)
            attempt += 1

        return "Error: Failed to get response from LLM after multiple attempts."
    
    
    def run(self,
            prompt: str,
            top_p: float = 0.8,
            temperature: float = 0.7,
            max_length: int = 24576,
            max_try: int = 5):
        
        success = False
        try_times = 0

        while try_times < max_try:
            response = self.call_llm(
                messages=prompt,
                top_p=top_p,
                temperature=temperature,
                max_length=max_length,
            )
            
            if response != "Error: Failed to get response from LLM after multiple attempts.":
                success = True
                break
            else:
                try_times += 1
        
        return response, success