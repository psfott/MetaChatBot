import time

from base_chat import BaseChatController


class FastChatController(BaseChatController):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.headers = {
            "Content-Type": "application/json",
        }
        self.url = "http://127.0.0.1:8012/v1/chat/completions"


if __name__ == "__main__":
    fast = FastChatController("vicuna-7b-v1.5")
    while True:
        prompt = input("Input:")
        start = time.time()
        fast.chat(prompt)
        print(time.time() - start)
