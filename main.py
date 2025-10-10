import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, Future
from multiprocessing import Pool, freeze_support, Manager
from shutil import copyfile
from threading import Thread
from time import time, sleep
from typing import Union, Tuple

import keyboard
import psutil
import pydub.playback as playback
from pydub import AudioSegment

import global_data
from audio2face import Audio2FaceController
from azure_speech import SpeechController
from chat_glm import ChatGLMControllerLocal, ChatGLMController
from emotion_classifier import EmotionClassifier
from global_data import AUDIO_ROOT_PATH, SPEECH2GESTURE_MODEL_PATH, GESTURE_PATH, MULITI_THREAD, MULITI_PROCESS, \
    MAIN_THREAD, SPEECH_ONLY, SPEECH_AND_EMOTION, ALL, SPEECH_AUDIO_FILE, SPEECH_MIC, NO_SPEECH, SPEECH_ALL_FILE, \
    DEFAULT, BREAK, ORIGIN, AUDIO2FACE_PATH
from llama_cpp_chat import LlamaChatController
from thread_server import Server
from speech2gesture.audio.audio_files import read_wavfile
from speech2gesture.gesture_generator import GestureGenerator

global_data.init()


# 第一次建立TCP链接时间长,预先建立好链接
def prepare(content):
    speech_controller.synthesis({}, content, f"{AUDIO_ROOT_PATH}/prepare.wav", is_streaming=USE_STREAMING)
    if USE_SPEECH != SPEECH_ALL_FILE:
        gesture_generator.generate(f"{AUDIO_ROOT_PATH}/prepare.wav", f"{GESTURE_PATH}/prepare.bvh",
                                   first_pose=f"{GESTURE_PATH}/prepare.bvh", verbose_level=True)
    audio2face_controller.set_livelink()
    if not USE_STREAMING:
        audio2face_controller.set_track("prepare.wav")
        audio2face_controller.play()
    if USE_SPEECH != SPEECH_ALL_FILE:
        global_data.get("process_pool").join()
        chat_controller.test()


def play(track_name, answer_index: int, emotion_dict, audio_length: float):
    # audio2face_controller.set_livelink()
    audio2face_time = time()
    audio2face_controller.set_emotions(emotion_dict)
    audio2face_controller.set_track(track_name)
    audio2face_controller.play()
    global gesture_start_pos
    gesture_start_pos[answer_index] = time()
    server.send_data(f"Audio length:{audio_length:.4f}")
    print("Audio2face api time {:.2f}".format(time() - audio2face_time))


def threaded_play(track_name, answer_index, emotion_dict, audio_length):
    while str(server.done_index) not in track_name:
        sleep(0.01)
    play(track_name, answer_index, emotion_dict, audio_length)


def async_speech(speech_controller_param=None, emotion_dict=None, answer="", answer_index=1, audio_flag_param=None):
    if audio_flag_param is None:
        audio_flag_param = []
    if emotion_dict is None:
        emotion_dict = dict()
    speech_time = time()
    speech_controller_param = SpeechController() if speech_controller_param is None else speech_controller_param
    if USE_STREAMING:
        if answer_index == 1 or audio_flag_param[answer_index - 1]:
            print("synthesize")
            emotion, audio_length = speech_controller_param.synthesis(emotion_dict, answer,
                                                                      f"{AUDIO_ROOT_PATH}/answer{answer_index}.wav",
                                                                      is_streaming=USE_STREAMING,
                                                                      audio2face=audio2face_controller, server=server,
                                                                      playing_pos=gesture_start_pos,
                                                                      answer_index=answer_index)
    else:
        emotion, audio_length = speech_controller_param.synthesis(emotion_dict, answer,
                                                                  f"{AUDIO_ROOT_PATH}/answer{answer_index}.wav",
                                                                  is_streaming=USE_STREAMING)
    # if USE_SPEECH == NO_SPEECH:
    #     global global_background_label
    #     global audio_file_index
    #     global log
    #     copyfile(f"{ROOT_PATH}/answer{answer_index}.wav",
    #              f"{ROOT_PATH}/pre-generate/{global_background_label}{audio_file_index + 1}.wav")
    #     audio_file_index = audio_file_index + 1

    print("Generate speech{} time {:.2f}".format(answer_index, time() - speech_time))
    audio_flag_param[answer_index] = True
    return answer_index, audio_length, emotion


def speech_callback(future: Union[Future, Tuple]):
    if not USE_STREAMING:
        if isinstance(future, Future):
            answer_index, audio_length, emotion = future.result()
        else:
            answer_index, audio_length, emotion = future[0], future[1], future[2]

        if not USE_ASYNC_AUDIO_AND_GESTURE and server.condition == SPEECH_AND_EMOTION:
            server.send_data(f"Audio label:{emotion}")
            audio_play(answer_index, audio_length, emotion_dict_list)

        elif USE_ASYNC_AUDIO_AND_GESTURE:
            server.send_data(f"Audio label:{emotion}")
            audio_play(answer_index, audio_length, emotion_dict_list)
        else:
            pass

    global answer_length


def async_gesture(gesture_generator_param, emotion_dict: dict, first_pose: bool = None, answer_index=0,
                  audio_flag_param=None,
                  gesture_flag_param=None):
    first_pose_path = f"{GESTURE_PATH}/answer{answer_index - 1}.bvh" if first_pose and answer_index != 1 else None

    gesture_time = time()
    if isinstance(gesture_generator_param, GestureGenerator):
        audio_length, _ = gesture_generator_param.generate(f"{AUDIO_ROOT_PATH}/answer{answer_index}.wav",
                                                           f"{GESTURE_PATH}/answer{answer_index}.bvh",
                                                           first_pose=first_pose_path, audio_flag=audio_flag_param,
                                                           gesture_flag=gesture_flag_param,
                                                           emotion_dict=emotion_dict, verbose_level=1)
    else:
        gesture_generator_data = gesture_generator_param["generator"]
        gesture_generator = GestureGenerator(f"{SPEECH2GESTURE_MODEL_PATH}/options.json", seed=int(time()),
                                             use_preload=False)
        gesture_generator.import_model_and_animation(gesture_generator_data)
        audio_length, _ = gesture_generator.generate(f"{AUDIO_ROOT_PATH}/answer{answer_index}.wav",
                                                     f"{GESTURE_PATH}/answer{answer_index}.bvh",
                                                     first_pose=first_pose_path, audio_flag=audio_flag_param,
                                                     gesture_flag=gesture_flag_param,
                                                     emotion_dict=emotion_dict, verbose_level=1)
        print("Generate gesture{} time {:.2f}".format(answer_index, time() - gesture_time))

    if audio_flag_param is not None and gesture_flag_param is not None:
        audio_flag_param[answer_index] = False

        if answer_index != 1:
            gesture_flag_param[answer_index - 1] = False
        else:
            gesture_flag_param[answer_index] = False

    return answer_index, audio_length


def gesture_callback(future):
    if isinstance(future, Future):
        answer_index, audio_length = future.result()
    else:
        answer_index, audio_length = future[0], future[1]

    gesture_flag[answer_index] = True

    if USE_ASYNC_AUDIO_AND_GESTURE:
        global gesture_start_pos
        now = time()
        gesture_start_pos[answer_index] = now - gesture_start_pos[answer_index] if now > gesture_start_pos[
            answer_index] and gesture_start_pos[answer_index] != 0.0 else 0.0
        server.send_data(f"Gesture pos:{answer_index} {gesture_start_pos[answer_index]}")
        sleep(0.1)

    server.send_data(f"Gesture index:{answer_index}")

    if not USE_ASYNC_AUDIO_AND_GESTURE:
        global emotion_dict_list
        audio_play(answer_index, audio_length, emotion_dict_list)


def audio_play(answer_index, audio_length, emotion_dict_list_param):
    global start_time
    if server.condition == SPEECH_ONLY:  # 清除表情
        for emotion in emotion_dict_list_param:
            for key in emotion.keys():
                emotion[key] = 0.0

    if time() - start_time <= 3.0:
        sleep(5 - time() + start_time)  # 反应时间短则继续等待

    if answer_index == 1:
        play(f"answer{answer_index}.wav", answer_index, emotion_dict_list_param[answer_index - 1], audio_length)
    else:
        Thread(target=threaded_play,
               args=(f"answer{answer_index}.wav", answer_index, emotion_dict_list_param[answer_index - 1],
                     audio_length,)).start()


def chat(use_speech=NO_SPEECH, processor_type: int = 1, use_first_pose=False, is_blocking=False,
         use_async_gesture_and_audio=False):
    global global_background_label, global_background_index
    global audio_file_index
    global log
    global start_time

    start_time = time()

    if use_speech == SPEECH_MIC:
        prompt, recognize_time = speech_controller.recognize()
        print("Recognize speech time {:.2f}, (including speech time)".format(recognize_time))
    elif use_speech == SPEECH_AUDIO_FILE:
        global audio_file
        global audio_text
        prompt, recognize_time = audio_text, 1
        print(f"Recognized: {audio_text}")
        sound = AudioSegment.from_file(audio_file, format="wav")
        playback.play(sound)
    elif use_speech == SPEECH_ALL_FILE:
        global voice_type
        audio_files = os.listdir("./audio/output/pre-generate")
        bvh_files = os.listdir("./gesture/pre-generate")
        if voice_type == DEFAULT:
            audio_files = [audio_file for audio_file in audio_files if
                           global_background_label in audio_file and "-" not in audio_file and str(
                               global_background_index) in audio_file]
            bvh_files = [bvh_file for bvh_file in bvh_files if
                         global_background_label in bvh_file and "-" not in bvh_file and str(
                             global_background_index) in bvh_file]
        else:
            suffix = "-break" if voice_type == BREAK else "-origin"
            audio_files = [audio_file for audio_file in audio_files if
                           global_background_label in audio_file and suffix in audio_file and str(
                               global_background_index) in audio_file]
            bvh_files = [bvh_file for bvh_file in bvh_files if
                         global_background_label in bvh_file and suffix in bvh_file and str(
                             global_background_index) in bvh_file]

        server.send_data(f"Answer length and flag:{len(audio_files)} {int(use_async_gesture_and_audio)}")  # 发送总长度
        sleep(0.1)
        for file_index, file_tuple in enumerate(zip(audio_files, bvh_files)):
            audio_file = file_tuple[0]
            bvh_file = file_tuple[1]
            copyfile(f"./gesture/pre-generate/{bvh_file}", f"./gesture/answer{file_index + 1}.bvh")
            fs, audio_data = read_wavfile(
                f"./audio/output/pre-generate/{audio_file}",
                rescale=True,
                desired_fs=16000,
                desired_nb_channels=None,
                out_type="float32",
                logger=None,
                need_length=True
            )
            audio_length = len(audio_data) / fs
            server.send_data(f"Gesture index:{file_index + 1}")
            play(f"{audio_file}", file_index + 1, chat_controller.generate_label_emotion_dict(global_background_label),
                 audio_length)
            sleep(audio_length)
        return
    else:
        prompt = input("Prompt:")
        if prompt.lower() == "q" or prompt.lower() == "quit":
            raise KeyboardInterrupt
        if global_background_label:
            speech_controller.synthesis("Default", prompt,
                                        f"{AUDIO_ROOT_PATH}/input/{global_background_label}{audio_file_index + 1}.wav",
                                        is_streaming=USE_STREAMING)
            log.write(f"Q: {prompt}\n")
        recognize_time = 1

    global background_label, background_index, last_background_index, last_background_label
    background_label = server.background_label
    background_index = server.background_index

    if background_label != "":
        if background_index == last_background_index and background_label == last_background_label:
            pass
        else:
            last_background_label = background_label
            last_background_index = background_index
            print(f"Background {background_label}{background_index} set!")
            chat_controller.set_background(background_label, background_index)
            # if USE_FAKE_SPEECH:
            #     chat_controller.load_background_dialog(background_label, background_index)

    if recognize_time != 0:
        recognize_time = time()

        if background_label:
            answers, temp_emotion_dict_list = chat_controller.chat(prompt.strip(), use_emotion=True, check_label=True)
        else:
            answers, temp_emotion_dict_list = chat_controller.chat(prompt.strip(), use_emotion=True, check_label=True)

       # print(answers)

        if USE_SPEECH == NO_SPEECH:
            if global_background_label:
                output_answer = ""
                for sentence in answers:
                    output_answer += sentence
                log.write(f"A: {output_answer}\n")

        global emotion_dict_list
        emotion_dict_list = temp_emotion_dict_list

        future_speech = []
        future_gesture = []

        print("Generate response time {:.2f}".format(time() - recognize_time))

        global answer_length
        answer_length = len(answers)

        global gesture_start_pos
        gesture_start_pos = [0.0 for i in range(6)]

        server.send_data(f"Answer length and flag:{answer_length} {int(use_async_gesture_and_audio)}")  # 发送总长度
        sleep(0.1)
        server.send_data(f"Streaming:{int(USE_STREAMING)}")

        for answer_index, answer_emotion_tuple in enumerate(zip(answers, emotion_dict_list)):
            emotion_dict = answer_emotion_tuple[1]
            answer = answer_emotion_tuple[0]
            audio_length = 0.0
            if processor_type == MULITI_THREAD:
                print("dddd")
                speech_task = speech_pool.submit(async_speech, speech_controller, emotion_dict, answer,
                                                 answer_index + 1,
                                                 audio_flag)
                if not is_blocking and use_async_gesture_and_audio:
                    speech_task.add_done_callback(speech_callback)
                elif server.condition == SPEECH_AND_EMOTION and not use_async_gesture_and_audio:
                    speech_task.add_done_callback(speech_callback)
                future_speech.append(speech_task)
            elif processor_type == MULITI_PROCESS:
                if not is_blocking and use_async_gesture_and_audio:
                    speech_task = speech_pool.apply_async(func=async_speech,
                                                          args=(
                                                              None,
                                                              emotion_dict, answer, answer_index + 1, audio_flag),
                                                          callback=speech_callback)
                elif server.condition == SPEECH_AND_EMOTION and not use_async_gesture_and_audio:
                    speech_task = speech_pool.apply_async(func=async_speech,
                                                          args=(
                                                              None,
                                                              emotion_dict, answer, answer_index + 1, audio_flag),
                                                          callback=speech_callback)
                else:
                    speech_task = speech_pool.apply_async(func=async_speech,
                                                          args=(
                                                              None,
                                                              emotion_dict, answer, answer_index + 1, audio_flag))
                future_speech.append(speech_task)
            else:
                start = time()
                speech_controller.synthesis(emotion_dict, answer, f"{AUDIO_ROOT_PATH}/answer{answer_index + 1}.wav",
                                            is_streaming=USE_STREAMING)
                print("Generate speech{} time {:.2f}".format(answer_index, time() - start))

            if server.condition == ALL:
                if processor_type == MULITI_THREAD:
                    gesture_task = gesture_pool.submit(async_gesture, gesture_generator, emotion_dict, use_first_pose,
                                                       answer_index + 1,
                                                       audio_flag, gesture_flag)
                    if not is_blocking:
                        gesture_task.add_done_callback(gesture_callback)
                    future_gesture.append(gesture_task)
                elif processor_type == MULITI_PROCESS:
                    gesture_task = gesture_pool.apply_async(func=async_gesture,
                                                            args=(
                                                                gesture_generator_dict,
                                                                emotion_dict, use_first_pose,
                                                                answer_index + 1,
                                                                audio_flag, gesture_flag,),
                                                            callback=gesture_callback)
                    if not is_blocking:
                        gesture_task = gesture_pool.apply_async(func=async_gesture,
                                                                args=(
                                                                    gesture_generator_dict,
                                                                    emotion_dict, use_first_pose,
                                                                    answer_index + 1,
                                                                    audio_flag, gesture_flag,))
                    future_gesture.append(gesture_task)
                else:
                    first_pose_path = f"{GESTURE_PATH}/answer{answer_index}.bvh" if use_first_pose and answer_index + 1 != 1 \
                        else None
                    audio_length, _ = gesture_generator.generate(f"{AUDIO_ROOT_PATH}/answer{answer_index + 1}.wav",
                                                                 f"{GESTURE_PATH}/answer{answer_index + 1}.bvh",
                                                                 first_pose=first_pose_path,
                                                                 emotion_dict=emotion_dict, verbose_level=1)
                    server.send_data(f"Gesture index:{answer_index + 1}")
                if processor_type == MAIN_THREAD:
                    audio_play(answer_index + 1, audio_length, emotion_dict_list)

        if is_blocking and processor_type != MAIN_THREAD:
            result_dict = {}
            if server.condition == ALL:
                for future in future_gesture:
                    if processor_type:
                        answer_index, audio_length = future.result()
                    else:
                        future.get()
                        answer_index, audio_length = future[0], future[1]
                    server.send_data(f"Gesture index:{answer_index}")
                    gesture_flag[answer_index] = True
                    result_dict[answer_index] = audio_length

                for answer_index, audio_length in result_dict.items():
                    audio_play(answer_index, audio_length, emotion_dict_list)

    else:
        server.send_data("Fail:")
        print("Please Retry!")


def launch_audio2face():
    for proc in psutil.process_iter(['name']):
        if "kit.exe".lower() in proc.info['name'].lower():
            print("Audio2face has already launched!")
            return
    # raise SystemExit("Plase launch audio2face first!")
    print("Waiting for launch Audio2face!")
    process = subprocess.Popen(['cmd.exe', '/c', os.path.join(AUDIO2FACE_PATH, "audio2face_headless.bat")], shell=True,
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


if __name__ == '__main__':

    USE_SPEECH = NO_SPEECH  # 是否使用语音

    global_background_label = None
    global_background_index = 0

    lang = "zh-CN"

    start_time = 0

    USE_THREAD = MULITI_THREAD  # 多线程或者多进程
    IS_BLOCKING = True  # 是否阻塞
    USE_ASYNC_AUDIO_AND_GESTURE = False  # 是否异步播放动画和语音,和阻塞设置冲突
    USE_STREAMING = True  # 流式合成
    USE_FIRST_POSE = True  # 是否使用上一个作为第一个姿势

    MAX_LENGTH = 6
    USE_SINGLE_SENTENCE = False
    USE_SINGLE_LABEL = False  # chatcontroller相关参数

    USE_FAKE_SPEECH = False

    launch_audio2face()
    # 建立服务器
    server = Server("127.0.0.1", 8012)
    server.start()

    manager = Manager()

    answer_length = 0
    background_label = ""
    background_index = 0

    last_background_label = ""
    last_background_index = 0
    temperature = 0.2

    audio_file = ""
    audio_text = ""
    audio_text_list = []
    audio_file_list = []
    audio_file_index = 0

    log = None
    # 相关全局变量
    chat_controller = None
    gesture_generator = None
    voice_type = None
    # 初始化控制器
    if USE_SPEECH != SPEECH_ALL_FILE:

        gesture_generator = GestureGenerator(f"{SPEECH2GESTURE_MODEL_PATH}/options.json", seed=int(time()),
                                             use_thread=False, use_gpu=True,
                                             use_preload=True,
                                             is_blocking=True,
                                             load_size=1)
        audio2face_controller = Audio2FaceController(is_streaming=USE_STREAMING)
    else:
        chat_controller = EmotionClassifier()
        audio2face_controller = Audio2FaceController(f"{AUDIO_ROOT_PATH}/output/pre-generate")
        voice_type = ORIGIN

    speech_controller = SpeechController(lang)
    # fastchat_controller = FastChatController("vicuna-7b-v1.5")
    # chatgpt_controller = ChatGPTController("gpt-3.5-turbo")
    # chat_controller = LlamaChatController(f"{LLM_PATH}/Wizard-Vicuna-7B-Uncensored.Q5_K_M.gguf", 43, lang=lang,
    #                                       use_single_sentence=USE_SINGLE_SENTENCE, use_label=USE_SINGLE_LABEL,
    #                                       max_length=MAX_LENGTH, temperature=temperature)

    chat_controller = ChatGLMController("glm-4-flash-250414", api_key="9dc9748e9df943c69a35d834dd8183af.YA2cgQbxxpqtmQcg", lang="zh-CN")
    # chat_controller = ChatGLMControllerLocal("", "zh-CN")

    if lang == "en-US":
        prepare("hello!")
    else:
        prepare("您好！")

    if global_background_label and isinstance(chat_controller, LlamaChatController):
        chat_controller.set_background(global_background_label, global_background_index)

    if USE_THREAD != MULITI_PROCESS:
        audio_flag = [False for i in range(6)]
        emotion_dict_list = []
        gesture_flag = [False for i in range(6)]
        speech_pool = ThreadPoolExecutor(5)
        gesture_pool = ThreadPoolExecutor(5)

    else:
        speech_pool = Pool(5)
        gesture_pool = Pool(5)
        audio_flag = manager.list()
        emotion_dict_list = manager.list()
        gesture_flag = manager.list()
        for i in range(6):
            audio_flag.append(False)  # 音频是否合成完毕标志位
            gesture_flag.append(False)
        gesture_generator_dict = manager.dict()
        if gesture_generator is not None:
            data_dict = gesture_generator.export_model_and_animation()
            gesture_generator_dict["generator"] = data_dict

    gesture_start_pos = [0.0 for i in range(6)]

    freeze_support()

    if USE_SPEECH == SPEECH_AUDIO_FILE:
        audio_file_list = os.listdir("./audio/input")
        if global_background_label:
            with open(f"./log/{global_background_label}.txt", "r", encoding="utf-8") as file:
                text = file.readlines()
            audio_text_list = [line.split(":")[-1].strip() for line in text if "Q" in line]
            audio_file_list = [file for file in audio_file_list if global_background_label in file]
        audio_file_index = 0
    elif USE_SPEECH == NO_SPEECH:
        log = open(f"./log/{global_background_label}.txt", "w", encoding="utf-8")
    else:
        pass
    # noinspection PyBroadException
    try:
        while True:
            sleep(0.01)
            if server.speaking_flag:
                chat(SPEECH_MIC, USE_THREAD, USE_FIRST_POSE, IS_BLOCKING, USE_ASYNC_AUDIO_AND_GESTURE)
                server.speaking_flag = False

            if USE_SPEECH == SPEECH_MIC:
                if keyboard.is_pressed("t"):
                    chat(USE_SPEECH, USE_THREAD, USE_FIRST_POSE, IS_BLOCKING, USE_ASYNC_AUDIO_AND_GESTURE)
                elif keyboard.is_pressed("esc"):
                    break
                else:
                    pass
            elif USE_SPEECH == SPEECH_AUDIO_FILE:
                if keyboard.is_pressed("space"):
                    audio_file = "./audio/input/" + audio_file_list[audio_file_index]
                    audio_text = audio_text_list[audio_file_index]
                    chat(USE_SPEECH, USE_THREAD, USE_FIRST_POSE, IS_BLOCKING, USE_ASYNC_AUDIO_AND_GESTURE)
                    audio_file_index += 1
                    if audio_file_index == len(audio_file_list):
                        audio_file_index = 0
                        chat_controller.clear_messages()  # 重新录制
                elif keyboard.is_pressed("esc"):
                    break
                else:
                    pass
            elif USE_SPEECH == SPEECH_ALL_FILE:
                if keyboard.is_pressed("space"):
                    chat(USE_SPEECH, USE_THREAD, USE_FIRST_POSE, IS_BLOCKING, USE_ASYNC_AUDIO_AND_GESTURE)
            else:
                chat(USE_SPEECH, USE_THREAD, USE_FIRST_POSE, IS_BLOCKING, USE_ASYNC_AUDIO_AND_GESTURE)
    except Exception as e:
        print(e)
    finally:
        audio2face_controller.close()
        if isinstance(chat_controller, LlamaChatController):
            chat_controller.close()
        server.close()
        if log is not None:
            log.close()
        if USE_THREAD:
            speech_pool.shutdown()
            gesture_pool.shutdown()
        else:
            speech_pool.close()
            gesture_pool.close()
            speech_pool.join()
            gesture_pool.join()
        exit()
