import json
import gradio as gr
import LLM


# 假设我们有一个简单的chatbot模型函数
def chatbot(message, model_choice):
    model = LLM.LLM(model_name=model_choice)
    return model(message)

# 假设有一个简单的 agent 函数
def agent_function(message, model_choice):
    agent = LLM.Agent(model_name=model_choice)
    return json.dumps(agent(message))

# 创建左侧chatbot和右侧agent部分的接口
with gr.Blocks() as demo:
    # 创建两个部分的布局
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Chatbot")
            model_choice_chatbot = gr.Dropdown(["deepseek-r1", "deepseek-v3", "qwen-max"], label="Choose Model", value="deepseek-r1")
            message_input_chatbot = gr.Textbox(placeholder="Enter your message here...")
            response_output_chatbot = gr.Textbox(label="Chatbot_Response")
            message_input_chatbot.submit(chatbot, [message_input_chatbot, model_choice_chatbot], response_output_chatbot)

        with gr.Column():
            gr.Markdown("### Agent")
            model_choice_agent = gr.Dropdown(["deepseek-r1", "deepseek-v3", "qwen-max"], label="Choose Model", value="deepseek-r1")
            message_input_agent = gr.Textbox(label="Answer two Questions use '|' to split: 1. What is the object? 2.What is the material of your gripper?", placeholder="Enter your message here...")
            response_output_agent = gr.Textbox(label="Agent_Response")
            message_input_agent.submit(agent_function, [message_input_agent, model_choice_agent], response_output_agent)

# 启动Gradio界面
demo.launch()
