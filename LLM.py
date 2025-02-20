import json
import os
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

    def __call__(self, message, *args, **kwargs):
        message = self.format(message)
        message = json.loads(message)
        object = message["object"]
        response1 = json.loads(self.model1(message=str(message)))
        response2 = json.loads(self.model2(message=object))
        response3 = json.loads(self.model3(message=object))

        return {
            "factor": response1["factor"],
            "volume": response2["volume"],
            "density": response3["density"],
        }


if __name__ == "__main__":
    agent = Agent("deepseek-r1")
    print(agent("Apple|Rubber"))