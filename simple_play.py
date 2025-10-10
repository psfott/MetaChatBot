import json
import os
import shutil
from pathlib import Path
from socket import gethostname
from time import time, sleep

from audio2face import Audio2FaceController
from azure_speech import SpeechController
from emotion_classifier import EmotionClassifier
from global_data import SPEECH2GESTURE_MODEL_PATH, AUDIO_ROOT_PATH, GESTURE_PATH
from speech2gesture.audio.audio_files import convert_to_desired
from speech2gesture.gesture_generator import GestureGenerator, STREAMING_AND_FILE
from streaming_main import get_config, check_ports, launch_audio2face

import librosa


def get_audio_length(file_path):
    try:
        y, sr = librosa.load(file_path)
        duration = len(y) / sr
        return duration
    except Exception as e:
        print(f"Error: {e}")
        return None


def prepare(content):
    speech_synthesis_controller.synthesis("Default", content, f"{AUDIO_ROOT_PATH}/ted/prepare.wav", is_streaming=False)
    gesture_generator.generate(f"{AUDIO_ROOT_PATH}/ted/prepare.wav", f"{GESTURE_PATH}/ted/prepare.bvh",
                               verbose_level=True)
    audio2face_controller.set_livelink()


def get_text_and_emotion():
    text_content = {}
    emotion_content = {}

    if classifier is None:
        with open("log/ted/ted_emotion.json", 'rb') as f:
            chatgpt_emotions = json.load(f)

    with open("log/ted/ted.json", 'rb') as f:
        json_data = json.load(f)
        for emotion, texts in json_data.items():
            text_content[emotion] = {}
            emotion_content[emotion] = {}
            for text_list in texts:
                for idx, text in text_list.items():
                    text_content[emotion][idx] = "".join(text).strip()
                    print(text_content[emotion][idx])
                    if classifier is not None:
                        emotion_content[emotion][idx] = classifier.weighted_classify(text_content[emotion][idx])
                    else:
                        emotions = chatgpt_emotions[emotion][idx].copy()
                        if emotions["neutral"] >= 0.6:
                            for emotion_name in emotions.keys():
                                emotions[emotion_name] = 0.0
                        del emotions["neutral"]
                        emotion_content[emotion][idx] = emotions

    return text_content, emotion_content


if __name__ == "__main__":
    config = get_config(f"./config/playing_config_{gethostname()}.yaml")

    classifier = EmotionClassifier() if config["emotion_type"] == "EmotionClassifier" else None

    text_content, emotion_content = get_text_and_emotion()

    check_ports([config["server_port"]])

    # 建立服务器
    if config["server_type"] == "async":
        from async_server import Server
    else:
        from thread_server import Server

    server = Server("127.0.0.1", config["server_port"])
    server.start()

    launch_audio2face(config)

    gesture_generator = None

    root_path = os.path.join("F:\Audio2Face")
    audio2face_controller = Audio2FaceController(root_path=f"{root_path}", is_streaming=False)

    audio2face_controller.set_root_path(os.path.join(AUDIO_ROOT_PATH, "ted"))

    speech_synthesis_controller = SpeechController("en-US")

    playing_label = ""
    playing_type = ""
    audio2face_controller.set_livelink()

    total_audio_length = {}

    for file in os.listdir(f"{AUDIO_ROOT_PATH}/ted/"):
        if file.lower().endswith(".wav"):
            total_audio_length[Path(file).stem] = get_audio_length(f"{AUDIO_ROOT_PATH}/ted/{file}")
            convert_to_desired(f"{AUDIO_ROOT_PATH}/ted/{file}", desired_fs=48000, desired_nb_channels=1)

    print(f"total_audio_length: {total_audio_length}")
    try:
        while True:
            playing_label = server.background_label
            playing_type = server.playing_type
            if server.is_setting_background:
                for idx in range(1, len(text_content[playing_label]) + 1):

                    emotion = emotion_content[playing_label][str(idx)]
                    audio_file = Path(
                        os.path.join(f"{AUDIO_ROOT_PATH}/ted/", f"{playing_label}{idx}-{playing_type}.wav"))

                    if playing_type == "origin":
                        if not os.path.exists(audio_file):
                            raise FileNotFoundError(f"{audio_file} is not exist!")
                    else:
                        if not os.path.exists(audio_file) or config["force_generate"]:
                            speech_synthesis_controller.synthesis(
                                emotion,
                                text_content[playing_label][str(idx)],
                                str(audio_file), is_streaming=False
                            )
                            total_audio_length[audio_file.stem] = get_audio_length(str(audio_file))
                            convert_to_desired(str(audio_file), desired_fs=48000, desired_nb_channels=1)

                    bvh_file = f"{GESTURE_PATH}/ted/{audio_file.stem}.bvh"

                    if config["force_generate"] or not os.path.exists(bvh_file):

                        if os.path.exists(bvh_file.replace("synthesized", "origin")):
                            shutil.copy(bvh_file.replace("synthesized", "origin"), bvh_file)

                        if gesture_generator is None:
                            gesture_generator = GestureGenerator(f"{SPEECH2GESTURE_MODEL_PATH}/options.json",
                                                                 seed=int(time()),
                                                                 use_thread=False, use_gpu=True,
                                                                 use_preload=True,
                                                                 is_blocking=True, generating_mode=STREAMING_AND_FILE,
                                                                 load_size=1)
                        gesture_generator.generate(str(audio_file).replace("synthesized", "origin"), bvh_file,
                                                   emotion_dict=emotion,
                                                   verbose_level=True)

                    shutil.copy(bvh_file, f"{GESTURE_PATH}/answer1.bvh")

                    audio_length = total_audio_length[audio_file.stem]

                    audio2face_controller.set_track(audio_file.name)
                    audio2face_controller.set_emotions(emotion)
                    server.send_data(f"Answer length and flag:{1} {1}")
                    sleep(0.1)
                    server.send_data(f"Gesture index:{1}")
                    sleep(0.1)
                    server.send_data(f"Audio length:{audio_length:.2f}")
                    audio2face_controller.play()

                    print(f"Playing {audio_file}...")
                    sleep(audio_length + 0.5)

                server.is_setting_background = False
            sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")
