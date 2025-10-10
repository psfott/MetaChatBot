import os

import nltk
from cnsenti import Emotion
from transformers import pipeline
from global_data import US_classifier_emotion_map, background_emotion_map, CN_classifier_emotion_map


# nltk.download('punkt')

class EmotionClassifier:
    def __init__(self, lang="en-US"):
        if lang == "en-US":
            self.set_proxy()
            self.__classifier = pipeline("text-classification", model="michellejieli/emotion_text_classifier",
                                         top_k=5)
        else:
            self.__classifier = Emotion().emotion_count
        self.lang = lang
        self.__empty_emotion_map = {
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
        print("Classifier inited!")

    def set_proxy(self):
        proxy_url = 'http://127.0.0.1'
        proxy_port = '7890'

        # Set the http_proxy and https_proxy environment variables
        os.environ['http_proxy'] = f'{proxy_url}:{proxy_port}'
        os.environ['https_proxy'] = f'{proxy_url}:{proxy_port}'

    def generate_label_emotion_dict(self, label):
        emotion_dict = self.__empty_emotion_map.copy()
        if label == "neutral":
            del emotion_dict["neutral"]
            return emotion_dict
        else:
            emotion_dict[background_emotion_map[label]] = 1.0
        return emotion_dict

    def __emotion_map(self, scores, weight=1.0, weighted_emotion_dict=None):
        if weighted_emotion_dict is None:
            weighted_emotion_dict = {}
        if self.lang == "en-US":
            for score in scores[0]:
                emotion_dict = {}
                emotion = US_classifier_emotion_map[score["label"]]
                emotion_dict[emotion] = score["score"]
                weighted_emotion_dict[emotion] += weight * emotion_dict[emotion]
        else:
            sum = 0
            except_list = ["words", "sentences", "好"]
            for emotion in scores.keys():
                if emotion not in except_list:
                    sum += scores[emotion]
            if sum == 0:
                weighted_emotion_dict["neutral"] = 1.0
                return
            for emotion in scores.keys():
                if emotion not in except_list:
                    weighted_emotion_dict[CN_classifier_emotion_map[emotion]] = weight * scores[emotion] / sum

    def weighted_classify(self, content: str):

        weighted_emotion_dict = self.__empty_emotion_map.copy()

        total_len = len(content)

        sentences = nltk.sent_tokenize(content)

        for sentence in sentences:
            scores = self.__classifier(sentence)
            weight = len(sentence) / total_len
            self.__emotion_map(scores, int(weight), weighted_emotion_dict)

        if weighted_emotion_dict["neutral"] >= 0.6:  # 如果主导情绪为自然
            for emotion in weighted_emotion_dict.keys():
                weighted_emotion_dict[emotion] = 0.0
        del weighted_emotion_dict["neutral"]

        return weighted_emotion_dict

    def plain_classify(self, content):

        emotion_dict = self.__empty_emotion_map.copy()

        scores = self.__classifier(content)
        self.__emotion_map(scores, 1, emotion_dict)

        if emotion_dict["neutral"] >= 0.6:  # 如果主导情绪为自然
            for emotion in emotion_dict.keys():
                emotion_dict[emotion] = 0.0
        del emotion_dict["neutral"]

        return emotion_dict

    def check_label(self, answer, label):

        emotion_dict = self.__empty_emotion_map.copy()

        scores = self.__classifier(answer)
        self.__emotion_map(scores, 1, emotion_dict)

        if label in emotion_dict.keys():

            label_value = emotion_dict[label]

            sorted_emotion_dict = sorted(emotion_dict.items(), key=lambda d: d[1], reverse=True)[0:3]

            rank = -1
            # 找rank
            for index, item in enumerate(sorted_emotion_dict):
                if abs(item[1] - label_value) < 1e-5:
                    rank = index + 1
                    break

            if rank >= 2:  # 如果情绪不是第一
                print(f"Label {label} inconsistent.")
                emotion_dict[label] = sorted_emotion_dict[0][1] + 0.1 if sorted_emotion_dict[0][1] + 0.1 <= 1.0 else 1.0
                for index, item in enumerate(sorted_emotion_dict):
                    if index != rank - 1:
                        emotion_dict[item[0]] = 0.1

        if emotion_dict["neutral"] >= 0.55:  # 如果主导情绪为自然
            for emotion in emotion_dict.keys():
                emotion_dict[emotion] = 0.0
        del emotion_dict["neutral"]

        return emotion_dict


if __name__ == "__main__":
    emotion_classifier = EmotionClassifier()
    print(emotion_classifier.plain_classify(
        "We can use our feelings towards the natural world in a more constructive way, alongside the knowledge and "
        "technology that helps us rewind nature. We can have a positive function in the ecosystem."))

    # emotion = Emotion()
    # test_text = '我好开心啊，非常非常非常高兴！今天我得了一百分，我很兴奋开心，愉快，开心'
    # result = emotion.emotion_count(test_text)
    # print(result)
