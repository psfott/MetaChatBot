import os

azuresSpeech_emotion_map = {"sad": "sadness", "angry": "anger", "terrified": "fear", "cheerful": "joy",
                            "unfriendly": "disgust",
                            "excited": "amazement",
                            "Default": "neutral"}

reversed_azureSpeech_emotion_map = {v: k for k, v in azuresSpeech_emotion_map.items()}

US_classifier_emotion_map = {"sadness": "sadness", "anger": "anger", "fear": "fear", "joy": "joy",
                             "surprise": "amazement",
                             "disgust": "disgust",
                             "neutral": "neutral"}

reversed_US_classifier_emotion_map = {v: k for k, v in US_classifier_emotion_map.items()}

CN_classifier_emotion_map = {"乐": "joy", "怒": "anger", "惧": "fear", "哀": "sadness",
                             "惊": "amazement",
                             "恶": "disgust"}

reversed_CN_classifier_emotion_map = {v: k for k, v in CN_classifier_emotion_map.items()}

gesture_emotion_map = {"neutral": "neutral", "sad": "sadness", "happy": "joy", "angry": "anger",
                       "scared": "fear",
                       "sneaky": "disgust",
                       "laughing": "amazement",
                       }

reversed_gesture_emotion_map = {v: k for k, v in gesture_emotion_map.items()}

background_emotion_map = {"neutral": "neutral", "surprised": "amazement", "disgust": "disgust", "scared": "fear",
                          "angry": "anger", "sad": "sadness", "happy": "joy"}

BASE_PATH = os.getcwd()
AUDIO_ROOT_PATH = os.path.join(BASE_PATH, "audio")
SPEECH2GESTURE_MODEL_PATH = os.path.join(BASE_PATH, "model")
GESTURE_PATH = os.path.join(BASE_PATH, "gesture")
LLM_PATH = "F:/Model/hub/llama.cpp/"

AZURE_SPEECH_KEY = os.environ.get("AZURE_SPEECH_KEY", "")
AZURE_SERVE_REGION = os.environ.get("AZURE_SERVE_REGION", "southeastasia")

# 根据实际位置修改
AUDIO2FACE_PATH = "D:/Omniverse/pkg/deps/3757f0f707549e4cd2e0c789de84c010/"

GLOBALS_DICT = {}
MAIN_THREAD = 0
MULITI_THREAD = 1
MULITI_PROCESS = 2

SPEECH_ONLY = 1
SPEECH_AND_EMOTION = 2
ALL = 3

NO_SPEECH = 0
SPEECH_MIC = 1
SPEECH_AUDIO_FILE = 2
SPEECH_ALL_FILE = 3

NO_GESTURE = 0
GESTURE_TOTAL = 2
GESTURE_EACH = 1

DEFAULT = 1
BREAK = 2
ORIGIN = 3


def init():
    """在主模块初始化"""
    global GLOBALS_DICT
    GLOBALS_DICT = {}


def set(name, value):
    """设置"""
    try:
        GLOBALS_DICT[name] = value
        return True
    except KeyError:
        return False


def get(name):
    """取值"""
    try:
        return GLOBALS_DICT[name]
    except KeyError:
        return "Not Found"
