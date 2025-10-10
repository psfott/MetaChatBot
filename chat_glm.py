import os
import subprocess

import colorama

from base_chat import BaseChatController
from global_data import LLM_PATH


class ChatGLMController(BaseChatController):
    def __init__(self, model_name, lang="en-US", api_key="", use_classifier=True):
        super().__init__(model_name, lang, use_classifier)
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"


class ChatGLMControllerLocal(BaseChatController):
    def __init__(self, model_name, lang="en-US", port="8000", use_classifier=True, model_path=None):
        super().__init__(model_name, lang, use_classifier)
        self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
        self.port = port
        self.model_path = model_path
        self.__run()

    def __run(self):
        command = [
            "uvicorn",
            "chatglm_cpp.openai_api:app",
            "--host", "127.0.0.1",
            "--port", str(self.port)  # 将端口号转换为字符串
        ]

        if self.model_path is None:
            os.environ["MODEL"] = f"{LLM_PATH}chatglm-ggml.bin"
        else:
            os.environ["MODEL"] = self.model_path

        process = subprocess.Popen(command,
                                   shell=True,
                                   encoding="utf-8",
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   )

        for line in process.stderr:
            print(line.strip())
            if "Uvicorn" in line:
                print("LLM server is running...")
                break


if __name__ == "__main__":
    chat_controller = ChatGLMController("glm-4-flash-250414", api_key="9dc9748e9df943c69a35d834dd8183af.YA2cgQbxxpqtmQcg", lang="zh-CN")

    # from openai import OpenAI
    #
    # client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="qweqe")

    while True:
        prompt = input("Prompt:")

        response = chat_controller.chat(prompt, True)
        print(response)
        # start = time()
        # for text, emotion in response:
        #     print(text)
        #     print(emotion)

    # pipeline = chatglm_cpp.Pipeline(r"F:\Model\hub\llama.cpp\chatglm-ggml.bin")
    # start = time()
    # res = pipeline.chat([chatglm_cpp.ChatMessage(role="user", content="您好！我今天很开心！")])
    # print(res)
    # print(time() - start)
    # chatgpt_controller = ChatGLMController("glm-4", "zh-CN", True)
    # prompt = "您好！我今天很开心！"
    # start = time()
    # chatgpt_controller.chat(prompt, use_emotion=True, use_prompt=False)
    # print(time() - start)
