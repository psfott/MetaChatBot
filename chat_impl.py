import json
import os
import re
from abc import abstractmethod

import nltk


class ChatImplement:
    def __init__(self, lang="en-US"):
        self.background = {}
        self.txt_background = {}
        self.messages = []
        self.__read_background_from_txt()
        self.lang = lang

    @abstractmethod
    def chat(self, content, use_emotion=True, use_prompt=False):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def _extract_answer_and_label(self, answer, use_single_label=False, is_single_sentence=False):

        pattern = re.compile(r"(.+?)\[(.+?)\]")
        matches = pattern.findall(answer)
        labels = []
        answers = []
        answer_without_label = ""

        for match in matches:
            answers.append(match[0].strip())
            answer_without_label += match[0].strip() + " "
            labels.append(match[1].strip())

        answer_without_label = answer_without_label.strip() if matches else answer  # 针对无标签情况处理

        if use_single_label:
            answers.clear()
            answers.append(answer_without_label)
            labels = labels[0:1]
            return answers, labels

        if not is_single_sentence:
            return answers, labels
        else:
            if answer_without_label:
                answers = nltk.sent_tokenize(answer_without_label)  # 切分句子
            else:
                answers = nltk.sent_tokenize(answer)  # 切分句子
            return answers, labels

    def __read_background_from_txt(self):
        with open("prompt/background.json", "r", encoding="utf8") as f:
            data = json.load(f)
        for key, value in data.items():
            background_answer = ""
            for answer in value["answers"]:
                background_answer += answer
            self.txt_background[key] = {"question": value["questions"], "answer": background_answer}

    def clear_messages(self):
        self.messages.clear()

    def set_background(self, label, question_index=0):
        question = ""

        prompts = list((f"Please try best to talk about his experience and generate response with {label} emotion."
                        "And please label one of the seven emotion tag",
                        "(neutral, sadness, joy, anger, fear, disgust, amazement) at the end of each sentence",
                        "Here is an example of your response: "
                        "\"I'm sorry to hear that.[sadness] How can I help you?[neutral]\""))

        if label in self.txt_background.keys():
            background = self.txt_background[label]
            question = background["question"][question_index]
            self.background["answer"] = background["answer"]
        else:
            prompts[0] = prompts[0].replace(f" {label} ", " ")
            self.background["answer"] = "Sure, I'll do that. [neutral]"

        for prompt in prompts:
            question += prompt

        self.background["question"] = question

        self.clear_messages()
