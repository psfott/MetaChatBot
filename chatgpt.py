import time

from base_chat import BaseChatController


class ChatGPTController(BaseChatController):
    def __init__(self, model_name, api_key="", use_classifier=True):
        super().__init__(model_name, use_classifier)
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.url = "https://api.openai.com/v1/chat/completions"


if __name__ == "__main__":
    chatgpt_controller = ChatGPTController("gpt-3.5-turbo", False)
    while True:
        prompt = input(">>")
        start = time.time()
        chatgpt_controller.chat(prompt, use_emotion=True, use_prompt=True)
        print(time.time() - start)
