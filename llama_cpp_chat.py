import keyboard
from llama_cpp import Llama
from pydub import AudioSegment, playback

from azure_speech import SpeechController
from chat_impl import ChatImplement
from emotion_classifier import EmotionClassifier
from global_data import AUDIO_ROOT_PATH, LLM_PATH


# layers为35大约4GB显存


class LlamaChatController(ChatImplement):

    def __init__(self, model_path, layers, lang="en-US", use_single_sentence=False, use_label=True, max_length=6,
                 temperature=0.2):
        super().__init__(lang)

        self.llm = Llama(model_path=model_path, n_gpu_layers=layers, chat_format="chatml", n_ctx=1024)

        self.use_single_sentence = use_single_sentence
        self.use_label = use_label
        self.max_length = max_length
        self.temperature = temperature
        self.classifier = EmotionClassifier(lang)
        self.question_index = None
        self.is_fake_question = None
        self.fake_answers = []

    def chat(self, content, emotion_label=None, use_emotion=True, check_label=True, print_label=True):
        if self.background:
            self.messages.insert(0, {"role": "assistant", "content": self.background["answer"]})
            self.messages.insert(0, {"role": "user", "content": self.background["question"]})

        if self.is_fake_question:
            fake_question = self.fake_answers[self.question_index]
            self.messages.append({"role": "user", "content": fake_question})
            self.question_index = self.question_index + 1
            if self.question_index == len(self.fake_answers) - 1:
                self.question_index = 0

        emotion_dict_list = []

        self.messages.append({"role": "user", "content": content})
        response = self.llm.create_chat_completion(messages=self.messages, temperature=self.temperature)

        llamacpp_answer = response["choices"][0]["message"]["content"]

        answers, labels = self._extract_answer_and_label(llamacpp_answer, self.use_label, self.use_single_sentence)

        answers = [single_answer.strip() for single_answer in answers]  # 去除头尾空格，换行

        if emotion_label is None:
            if use_emotion:
                for single_answer, label in zip(answers, labels):
                    if check_label:
                        emotion_dict = self.classifier.check_label(single_answer, label)
                    else:
                        emotion_dict = self.classifier.plain_classify(single_answer)
                    emotion_dict_list.append(emotion_dict)
                if not emotion_dict_list:  # 对应没有标签的情况
                    emotion_dict_list.append(self.classifier.plain_classify(llamacpp_answer))
            else:
                emotion_dict_list.append(self.classifier.plain_classify(llamacpp_answer))
        else:
            emotion_dict = self.classifier.generate_label_emotion_dict(emotion_label)
            for i in range(len(answers)):
                emotion_dict_list.append(emotion_dict)

        if len(llamacpp_answer) < 300:
            self.messages.append({"role": "assistant", "content": llamacpp_answer})
        else:
            print(f"Answer is too long!")
            self.messages.pop()

        if self.background:
            self.messages = self.messages[2:]

        if len(self.messages) > self.max_length:
            self.messages = self.messages[2:]  # 删除前两个对话

        if print_label:
            print(f"Answers:{answers}")
            print(f"Labels:{labels}")
            print(f"Emotion dict:{emotion_dict_list}")

        llamacpp_answer = answers if answers else [llamacpp_answer]

        return llamacpp_answer, emotion_dict_list

    def test(self):
        test_chat = "hello!" if self.lang == "en-US" else "您好！"
        result = self.chat(test_chat)
        self.messages.clear()
        if result:
            print("Test success!")

    def close(self):
        self.messages.clear()

    def load_background_dialog(self, label, index):
        self.is_fake_question = True
        self.fake_answers = []
        self.question_index = 0
        with open(f"./log/experiment/{label}{index}.txt") as f:
            for line in f:
                if "Q:" in line:
                    self.fake_answers.append(line.split("Q:")[-1].strip())


if __name__ == "__main__":
    MAX_LENGTH = 6
    USE_SINGLE_SENTENCE = False
    USE_SINGLE_LABEL = True

    USE_FAKE_SPEECH = False

    TEXT = 0
    SPEECH = 1

    condition = SPEECH
    lang = "zh-CN"

    llamacpp_controller = LlamaChatController(f"{LLM_PATH}/Wizard-Vicuna-7B-Uncensored.Q5_K_M.gguf", 43, lang=lang,
                                              use_single_sentence=USE_SINGLE_SENTENCE, use_label=USE_SINGLE_LABEL,
                                              max_length=MAX_LENGTH, temperature=0.0)

    # emotion_prompt = ("In the following conversation, please analyze the possible emotion and intensity "
    #                   "for your response to the User"
    #                   "and return them to me in the following json format in your response."
    #                   "{response\":\"your response\", \"amazement\": 0.0, \"anger\": 0.0, "
    #                   "\"cheekiness\": 0.0, \"disgust\": 0.0, \"fear\": 0.0,"
    #                   "\"grief\": 0.0, \"joy\": 0.0,\"outofbreath\": 0.0, \"pain\": 0.0, \"sadness\": "
    #                   "0.0}")

    # answer, _ = test.chat(emotion_label_prompt, use_emotion=False)
    # label = "happy"
    # index = 2
    #
    # prompt = ""
    # llamacpp_controller.set_background(label, index)
    # if USE_FAKE_SPEECH:
    #     llamacpp_controller.load_background_dialog(label, index)

    speech_controller = SpeechController(lang)

    # with open(f"./log/experiment/{label}{index}.txt", "w", encoding="utf-8") as f:

    while True:
        answer = None
        emotion = None

        if condition == TEXT:
            prompt = input("Prompt:")
            if prompt.lower() == "quit":
                break
            answer, emotion = llamacpp_controller.chat(f"{prompt.strip()}", use_emotion=True,
                                                       check_label=True, print_label=False)
            print(answer[0])
            speech_controller.synthesis(emotion, answer,
                                        f"{AUDIO_ROOT_PATH}/test.wav")
            sound = AudioSegment.from_file(f"{AUDIO_ROOT_PATH}/test.wav", format="wav")
            playback.play(sound)
        else:
            if keyboard.is_pressed("space"):
                prompt, recognize_time = speech_controller.recognize()
                if recognize_time != 0:
                    answer, emotion = llamacpp_controller.chat(f"{prompt}", use_emotion=True,
                                                               check_label=True, print_label=False)
                    answer = answer[0]

                speech_controller.synthesis(emotion, answer,
                                            f"{AUDIO_ROOT_PATH}/test.wav")
                sound = AudioSegment.from_file(f"{AUDIO_ROOT_PATH}/test.wav", format="wav")
                playback.play(sound)
