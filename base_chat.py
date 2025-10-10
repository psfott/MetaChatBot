import ast
import json
import re

import requests

from chat_impl import ChatImplement
from emotion_classifier import EmotionClassifier

MAX_LENGTH = 4  # 上下文长度


class BaseChatController(ChatImplement):
    def __init__(self, model_name, lang="en-US", use_classifier=True):
        super().__init__(lang)
        self.session = requests.Session()
        self.url = None
        self.headers = None
        self.model = model_name
        if use_classifier:
            self.classifier = EmotionClassifier(lang)
        else:
            self.classifier = None

        if lang == "en-US":
            self.emotion_prompt = (
                "Please label each sentence of your reply in the following "
                "conversation with one of the seven emotions: "
                "neutral, sadness, joy, anger, fear, disgust, amazement."
            )
            self.emotion_prompt_answer = "Sure, I'll try my best to do that. [neutral]"

        elif lang == "zh-CN":
            self.emotion_prompt = ("请您在接下来回复中每句话的末尾标注上neutral, "
                                   "sadness, joy, anger, fear, disgust, amazement这七个情感标签中的一个。"
                                   "下面是您回复的例子："
                                   "你好！[happy] 很高兴认识你。[happy]")
            self.emotion_prompt_answer = "好的，我会尽力帮助你。[neutral]"
        else:
            pass

    def close(self):
        self.session.close()

    def _extract_dict(self, text):
        pattern = re.compile(r".*({.*}).*", re.S)
        match = pattern.match(text)
        if match is None:
            return {}
        else:
            res = pattern.match(text).group(1)
            return ast.literal_eval(res)

    def _post(self, prompt, messages=None, max_tokens=250, temperature=0.8, is_streaming=False):
        new_message = [{"role": "user", "content": f"{prompt}"}]
        if messages is None:
            messages = new_message

        data = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": is_streaming
        }

        return requests.post(self.url, headers=self.headers, json=data, stream=is_streaming)

    def _plain_chat(self, content, max_token=500):
        self.messages.append({"role": "user", "content": content})
        response = self._post(content, self.messages, max_token).json()
        answer = response["choices"][0]["message"]["content"]
        self.messages.append({"role": "assistant", "content": answer})
        if len(self.messages) > MAX_LENGTH:
            self.messages = self.messages[2:]  # 删除前两个对话
        return answer

    def _prompt_emotion_chat(self, content, max_token=500):

        self.messages.insert(0, {"role": "assistant", "content": "".join(self.emotion_prompt_answer)})
        self.messages.insert(0, {"role": "user", "content": "".join(self.emotion_prompt)})

        self.messages.append({"role": "user", "content": content})
        response = self._post(content, self.messages, max_token).json()
        answer = response["choices"][0]["message"]["content"]

        self.messages = self.messages[2:]  # 删除插入的元素两个对话
        self.messages.append({"role": "assistant", "content": answer})
        if len(self.messages) > MAX_LENGTH:
            self.messages = self.messages[2:]  # 删除前两个对话

        return answer

    def chat(self, content, use_emotion=True, use_prompt=False, check_label=True):

        if self.background:
            self.messages.insert(0, {"role": "assistant", "content": self.background["answer"]})
            self.messages.insert(0, {"role": "user", "content": self.background["question"]})

        emotion_dict_list = []
        answers = []

        if self.__class__ == "ChatGLMController":
            token = 2048
        else:
            token = 500

        if use_emotion:
            if use_prompt:
                answer = self._prompt_emotion_chat(content, token)
                if self.lang == "zh-CN":
                    chinese_match = re.search(r'[\u4e00-\u9fff](.*?)]', answer)
                    if chinese_match:
                        # 返回匹配到的第一个句子
                        answer = chinese_match.group(0)
            else:
                answer = self._plain_chat(content, token)
            answers, labels = self._extract_answer_and_label(answer, False)

            if self.classifier is not None:
                for single_answer, label in zip(answers, labels):
                    if check_label:
                        emotion_dict = self.classifier.check_label(single_answer, label)
                    else:
                        emotion_dict = self.classifier.plain_classify(single_answer)
                    emotion_dict_list.append(emotion_dict)

                if not emotion_dict_list:
                    emotion_dict_list.append(self.classifier.plain_classify(answer))
            else:
                emotion_map = {
                    "amazement": 0.0,
                    "anger": 0.0,
                    "cheekiness": 0.0,
                    "disgust": 0.0,
                    "fear": 0.0,
                    "grief": 0.0,
                    "joy": 0.0,
                    "outofbreath": 0.0,
                    "pain": 0.0,
                    "sadness": 0.0,
                    "neutral": 0.0  # 用于最后计算
                }  # 加权的结果
                for label in labels:
                    copy_emotion = emotion_map.copy()
                    if label == "neutral":
                        del copy_emotion["neutral"]
                    else:
                        copy_emotion[label] = 1.0
                    emotion_dict_list.append(copy_emotion)
        else:
            answer = self._plain_chat(content, token)

        if self.background:
            self.messages = self.messages[2:]

        if len(self.messages) > MAX_LENGTH:
            self.messages = self.messages[2:]  # 删除前两个对话

        print(f"Response:{answer}")
        print(f"Emotion:{emotion_dict_list}")

        answer = answers if answers else [answer]

        return answer, emotion_dict_list

    def stream_chat(self, content, use_emotion=True, min_length=10, end_simple=None):

        if end_simple is None:
            end_simple = [".", "?", "!", "。", "？", "!"]

        self.messages.append({"role": "user", "content": content})

        response = self._post(content, self.messages, is_streaming=True, max_tokens=2048)

        text_chunk = ""
        total_answer = ""

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    content = decoded_line[6:]  # Remove the "data: " prefix
                    chunk = json.loads(content)
                    data = chunk["choices"][0]
                    if "finish_reason" in data:
                        return

                    if 'delta' in data and "content" in data["delta"]:
                        text = data['delta']["content"]
                        if text.strip() == "":
                            continue

                        text_chunk = text_chunk + text
                        total_answer = total_answer + text

                        if text in end_simple and len(text_chunk) >= min_length:
                            if use_emotion:
                                emotion_dict = self.classifier.plain_classify(text_chunk)
                            else:
                                emotion_dict = {
                                    "amazement": 0.0,
                                    "anger": 0.0,
                                    "cheekiness": 0.0,
                                    "disgust": 0.0,
                                    "fear": 0.0,
                                    "grief": 0.0,
                                    "joy": 0.0,
                                    "outofbreath": 0.0,
                                    "pain": 0.0,
                                    "sadness": 0.0,
                                }  # 加权的结果
                            yield text_chunk, emotion_dict
                            text_chunk = ""

        self.messages.append({"role": "assistant", "content": total_answer})
        if len(self.messages) > MAX_LENGTH:
            self.messages = self.messages[2:]  # 删除前两个对话

    def test(self):
        result = self._plain_chat("hello!")
        print(self.classifier.plain_classify(result))
        if result:
            print("Test success!")
        self.messages.clear()
