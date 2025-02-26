import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
from openai import OpenAI

import prompt


class LLM:
    def __init__(self, model_name, prompt=prompt.prompt):
        self.model_name = model_name.lower()
        self.prompt = prompt.lower()
        self.client = OpenAI(
            # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
            api_key=os.getenv("API_KEY"),
            # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key  
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    
    def __call__(self, message:str, *args, **kwargs):
        prompt = self.prompt.lower()
        message = message.lower()
        completion = self.client.chat.completions.create(
            model=self.model_name,  # 此处以 deepseek-r1 为例，可按需更换模型名称。
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': message}
            ]
        )
        return completion.choices[0].message.content

class Agent:
    def __init__(self, model_name):
        self.format = LLM(model_name=model_name, prompt=prompt.prompt_format)
        self.model1 = LLM(model_name=model_name, prompt=prompt.prompt_factor)
        self.model2 = LLM(model_name=model_name, prompt=prompt.prompt_estimation)
        self.model3 = LLM(model_name=model_name, prompt=prompt.prompt_density)
        # 创建线程池
        self.executor = ThreadPoolExecutor(max_workers=3)
    
    def _call_model1(self, message):
        response = json.loads(self.model1(message=str(message)))
        return {"factor": response["factor"]}
    
    def _call_model2(self, object):
        response = json.loads(self.model2(message=object))
        return {"volume": response["volume"]}
    
    def _call_model3(self, object):
        response = json.loads(self.model3(message=object))
        return {"density": response["density"]}
    
    def __call__(self, message, *args, **kwargs):
        # 首先运行format模型获取结构化内容
        formatted_message = self.format(message)
        message_dict = json.loads(formatted_message)
        object = message_dict["object"]
        
        # 创建三个线程任务
        future_to_model = {
            self.executor.submit(self._call_model1, message_dict): "model1",
            self.executor.submit(self._call_model2, object): "model2",
            self.executor.submit(self._call_model3, object): "model3"
        }
        
        # 收集结果
        results = {}
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.update(result)
            except Exception as exc:
                print(f'{model_name} generated an exception: {exc}')
        
        return results

if __name__ == "__main__":
    print(datetime.datetime.now())
    agent = Agent("deepseek-r1")
    print(agent("Apple|Rubber"))
    print(datetime.datetime.now())