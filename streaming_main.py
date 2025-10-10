import re
from typing import Union

import yaml
import os
import shutil
import subprocess
import wave
from concurrent.futures import ThreadPoolExecutor
from ctypes import cast
from time import time, sleep

import grpc
import keyboard
import numpy as np
import psutil
from _ctypes import POINTER
from socket import gethostname
from comtypes import CLSCTX_ALL
from pycaw.api.endpointvolume import IAudioMeterInformation
from pycaw.utils import AudioUtilities

from audio2face import Audio2FaceController
from azure_speech import SpeechController
from chat_glm import ChatGLMControllerLocal, ChatGLMController
from chatgpt import ChatGPTController
from global_data import NO_SPEECH, AUDIO2FACE_PATH, SPEECH2GESTURE_MODEL_PATH, LLM_PATH, AUDIO_ROOT_PATH, GESTURE_PATH, \
    SPEECH_MIC, GESTURE_TOTAL
from llama_cpp_chat import LlamaChatController
from speech2gesture.gesture_generator import GestureGenerator, STREAMING_AND_FILE, FILE
from streaming_server import audio2face_pb2_grpc, audio2face_pb2
from whisper_cpp import WhisperCpp


def launch_audio2face(config):
    for proc in psutil.process_iter(['name']):
        if "kit.exe".lower() in proc.info['name'].lower():
            print("Audio2face has already launched!")
            return
    # raise SystemExit("Plase launch audio2face first!")
    print("Waiting for launch Audio2face!")
    audio2face_path = config["audio2face_path"] if "audio2face_path" in config.keys() else AUDIO2FACE_PATH

    process = subprocess.Popen(['cmd.exe', '/c', os.path.join(audio2face_path, "audio2face_headless.bat")], shell=True,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               text=True)
    while True:
        line = process.stdout.readline()
        # for line in process.stdout:
        print(line.strip())
        if "transport.server.http" in line.lower():
            print("Launch Audio2face!")
            break
        # stdout, stderr = process.communicate()  # 记录错误日志文件


def get_audio_playback_status():
    try:
        devices = AudioUtilities.GetSpeakers()
        meter = devices.Activate(
            IAudioMeterInformation._iid_, CLSCTX_ALL, None)
        meter_info = cast(meter, POINTER(IAudioMeterInformation))
        peak = meter_info.GetPeakValue()
        if peak > 0:
            return True
        return False
    except OSError as e:
        return False


def check_ports(ports: Union[list, str]):
    pattern = re.compile(r'(\w+)\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+(\d+\.\d+\.\d+\.\d+):(\d+)\s+(\w+)\s+(\w+)')

    if ports is str:
        ports = [ports]

    for port in ports:
        net_cmd = f"netstat -ano| findstr {port}"
        process = subprocess.Popen(net_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while True:
            line = process.stdout.readline().strip()
            if line != "":
                match = pattern.match(line)
                if match is not None:
                    src_port = match.group(3)
                    src_ip = match.group(2)
                    if src_port == str(port) and (src_ip == "127.0.0.1" or src_ip == "0.0.0.0"):
                        pid = match.group(7)
                        if pid != "0":
                            task_cmd = f"taskkill /f /pid {pid}"
                            process = subprocess.Popen(task_cmd, shell=True, stdout=subprocess.PIPE,
                                                       stderr=subprocess.PIPE,
                                                       text=True)
                            res = process.stdout.readline()
                            if res.startswith("SUCCESS"):
                                print(f"Port {src_port} conflict! Terminated task {pid}!")
                            else:
                                raise OSError(f"Port {src_port} conflict! Failed to terminate task {pid}!")
            else:
                break


def init_and_test_controller(config):
    launch_audio2face(config)

    mode = STREAMING_AND_FILE if config["is_streaming_chat"] else FILE

    gesture_generator = GestureGenerator(f"{SPEECH2GESTURE_MODEL_PATH}/options.json", seed=int(time()),
                                         use_thread=False, use_gpu=True,
                                         use_preload=True,
                                         is_blocking=True, generating_mode=mode,
                                         load_size=1)

    audio2face_controller = Audio2FaceController(root_path=config["audio2face_scene_path"], is_streaming=config["is_streaming_chat"])

    speech_synthesis_controller = SpeechController(config["lang"])

    speech_recognition = config["speech_recognition_type"]
    chat_type = config["chat_type"]

    speech_recognition_controller = speech_synthesis_controller if speech_recognition == "Azure Speech" else WhisperCpp(
        config["speech_recognition_model_path"], lang=config["lang"])

    if chat_type == "ChatGLMLocal":
        chat_controller = ChatGLMControllerLocal("", lang=config["lang"], model_path=config["model_path"],
                                                 port=config["llm_port"])
    elif chat_type == "ChatGLMOnline":
        chat_controller = ChatGLMController("glm-4", lang=config["lang"], api_key=config["chat_api_key"])
    elif chat_type == "VicunaLocal":
        chat_controller = LlamaChatController(f"{LLM_PATH}/Wizard-Vicuna-7B-Uncensored.Q5_K_M.gguf", 43,
                                              lang=config["lang"],
                                              )
    elif chat_type == "ChatGPT":
        chat_controller = ChatGPTController("gpt-3.5-turbo", api_key=config["chat_api_key"])

    else:
        raise ValueError("Unknown Chat Model!")

    def prepare(content):
        speech_synthesis_controller.synthesis("Default", content, f"{AUDIO_ROOT_PATH}/prepare.wav", is_streaming=True)
        gesture_generator.generate(f"{AUDIO_ROOT_PATH}/prepare.wav", f"{GESTURE_PATH}/prepare.bvh", verbose_level=True)
        audio2face_controller.set_livelink()
        chat_controller.test()

    if config["lang"] == "en-US":
        prepare("hello!")
    else:
        prepare("您好！")

    return gesture_generator, audio2face_controller, speech_synthesis_controller, speech_recognition_controller, chat_controller


def steam_chat():
    recognize_time = 0
    prompt = ""

    if server.speaking_flag:
        prompt, recognize_time = speech_recognition_controller.recognize()
        print("Recognize speech time {:.2f}, (including speech time)".format(recognize_time))
        server.speaking_flag = False

    if USE_SPEECH == NO_SPEECH:
        prompt = input("Please enter:")
        recognize_time = 1
    elif USE_SPEECH == SPEECH_MIC:
        if keyboard.is_pressed("t"):
            prompt, recognize_time = speech_recognition_controller.recognize()
            print("Recognize speech time {:.2f}, (including speech time)".format(recognize_time))
    else:
        raise ValueError(f"Not implemented value {USE_SPEECH}!")
        # 语音或文字输入

    if recognize_time != 0 and prompt.strip() != "":  # 识别成功
        if STREAMING_CHAT:
            stream_chat_sentences(prompt)
        else:
            stream_chat_one_sentence(prompt)

    sleep(0.1)


def stream_chat_sentences(prompt, use_first_emotion=True):
    chat_response = chat_controller.stream_chat(prompt.strip(), True)

    audio_idx = 1

    sample_rate = 48000
    buffer_size = 16000
    audio_buffer = bytes(buffer_size)

    audio_length_list.clear()
    audio2face_controller.set_livelink()

    gesture_tasks = []

    def generate_gesture(gesture_generator, server, temp_counts, gesture_emotion_dict):
        shutil.copy(f"{AUDIO_ROOT_PATH}/answer{audio_idx}.wav", f"{AUDIO_ROOT_PATH}/temp/audio_clip{audio_idx}.wav")
        first_pose = None if temp_counts == 1 else f"./gesture/temp/audio_clip{temp_counts - 1}.bvh"
        gesture_generator.generate(f"./audio/temp/audio_clip{temp_counts}.wav",
                                   f"./gesture/temp/audio_clip{temp_counts}.bvh",
                                   first_pose=first_pose, server=server,
                                   emotion_dict=gesture_emotion_dict)

    def convert(data):
        sampled_data = np.frombuffer(data, dtype=np.int16) / 32768.0
        return sampled_data.astype(np.float32)

    def make_generator():

        sleep_between_chunks = 0.04
        block_until_playback_is_finished = False

        nonlocal audio_idx

        for temp_audio_file in os.listdir("./audio/temp"):
            os.remove(os.path.join("./audio/temp", temp_audio_file))

        first_emotion_dict = None
        try:
            for sentence, emotion_dict in chat_response:
                print(f"{sentence}")
                speech_time = time()
                if first_emotion_dict is None:
                    first_emotion_dict = emotion_dict.copy()

                for mask_word, replace_word in config["mask_words"].items():
                    sentence = sentence.replace(mask_word, replace_word)

                if use_first_emotion:
                    _, audio_streaming = speech_synthesis_controller.generate_audio_streaming(first_emotion_dict,
                                                                                              sentence)
                    audio2face_controller.set_emotions(emotion_dict)
                else:
                    _, audio_streaming = speech_synthesis_controller.generate_audio_streaming(emotion_dict, sentence)
                    audio2face_controller.set_emotions(emotion_dict)

                with wave.open(f"{AUDIO_ROOT_PATH}/answer{audio_idx}.wav", "wb") as wav_file:

                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.setcomptype('NONE', "not compressed")

                    if audio_idx == 1:
                        start_marker = audio2face_pb2.PushAudioRequestStart(
                            samplerate=sample_rate,
                            instance_name="/World/audio2face/PlayerStreaming",
                            block_until_playback_is_finished=block_until_playback_is_finished,
                        )
                        yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)

                    filled_size = audio_streaming.read_data(audio_buffer)
                    wav_file.writeframes(audio_buffer)

                    if audio_idx == 1:
                        global initial_playing_time
                        initial_playing_time = time()
                        server.send_data(f"Answer length and flag:{1} {1}")  # 发送总长度
                        sleep(0.1)
                        server.send_data(f"Streaming:{1}")

                    yield audio2face_pb2.PushAudioStreamRequest(
                        audio_data=convert(audio_buffer).tobytes()
                    )
                    times = 1

                    while filled_size > 0:
                        sleep(sleep_between_chunks)
                        filled_size = audio_streaming.read_data(audio_buffer)
                        if filled_size == 0:
                            break
                        wav_file.writeframes(audio_buffer)
                        yield audio2face_pb2.PushAudioStreamRequest(
                            audio_data=convert(audio_buffer).tobytes()
                        )
                        times += 1

                    audio_length = buffer_size / sample_rate / 2 * times
                    audio_length_list.append(audio_length)
                    #
                    # audio_playing_length = audio_length - (time() - start_generating_time)
                    # server.send_data(f"Audio{audio_idx} playing length:{audio_playing_length:.4f}")

                print(f"Generate speech{audio_idx} time {time() - speech_time:.2f}")

                gesture_emotion_dict = first_emotion_dict if use_first_emotion else emotion_dict

                # gesture_tasks.append(gesture_pool.submit(generate_gesture, gesture_generator, server, audio_idx,
                                                         # gesture_emotion_dict))
                audio_idx += 1
        except Exception as e:
            print(e)
        finally:
            return

    with grpc.insecure_channel("localhost:50051") as channel:
        print("Channel created!")
        stub = audio2face_pb2_grpc.Audio2FaceStub(channel)
        audio2face_controller.set_livelink()
        request_generator = make_generator()
        print("Sending audio data...")
        response = stub.PushAudioStream(request_generator)
        if response.success:
            print(f"Success!")
        else:
            print(f"Error: {response.message}")

    # if get_audio_playback_status():
    global initial_playing_time
    if time() - initial_playing_time < sum(audio_length_list):
        sleep(sum(audio_length_list) - (time() - initial_playing_time) + 0.2)

    server.send_data(f"Answer length and flag:{0} {1}")
    sleep(0.1)
    server.send_data(f"Streaming:{0}")
    for task in gesture_tasks:
        if not task.done():
            print(f"Canceled Task {task}")
            task.cancel()


def stream_chat_one_sentence(prompt):
    response_time = time()
    answers, emotion_dict_list = chat_controller.chat(prompt.strip(), use_emotion=True, check_label=True)
    print(f"Response time: {time() - response_time:.2f}")

    server.send_data(f"Answer length and flag:{1} {1}")  # 发送总长度
    sleep(0.1)
    server.send_data(f"Streaming:{1}")

    synthesis_time = time()
    _, audio_length, playing_time = speech_synthesis_controller.synthesis(emotion_dict_list[0], answers[0],
                                                                          f"{AUDIO_ROOT_PATH}/answer1.wav",
                                                                          is_streaming=True,
                                                                          audio2face=audio2face_controller,
                                                                          server=server,
                                                                          gesture_generator=gesture_generator,
                                                                          answer_index=1)
    print(f"Synthesis time: {time() - synthesis_time:.2f}")

    # if get_audio_playback_status():
    now = time()
    if now - playing_time < audio_length:
        sleep(audio_length - now + playing_time + 0.1)

    server.send_data(f"Answer length and flag:{0} {1}")
    sleep(0.1)
    server.send_data(f"Gesture index:{0}")
    sleep(0.1)
    server.send_data(f"Streaming:{0}")


def get_config(file='./config/config.yaml'):
    with open(file, 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


if __name__ == "__main__":

    config = get_config(f"./config/chat_config_{gethostname()}.yaml")

    USE_SPEECH = config["speech_type"]  # 是否使用语音
    STREAMING_CHAT = config["is_streaming_chat"]

    initial_playing_time = 0  # 根据时间计算正在播放的音频
    audio_length_list = []  # 保存音频时长
    gesture_pool = ThreadPoolExecutor(1)  # 姿势合成线程池

    check_ports([config["server_port"], config["llm_port"]])

    # 建立服务器
    if config["server_type"] == "async":
        from async_server import Server
    else:
        from thread_server import Server

    server = Server("127.0.0.1", config["server_port"])
    server.start()

    gesture_generator, audio2face_controller, speech_synthesis_controller, speech_recognition_controller, chat_controller = init_and_test_controller(
        config)  # 初始化各类控制器并测试

    while True:
        try:
            steam_chat()
        except Exception as e:
            print(e)
            audio2face_controller.close()
            speech_synthesis_controller.close()
            speech_recognition_controller.close()
            server.close()
            gesture_pool.shutdown(wait=False)
            break
