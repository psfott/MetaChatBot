import os
import wave
from html import escape
from concurrent.futures import ThreadPoolExecutor
from time import time, sleep

from global_data import AZURE_SPEECH_KEY, AZURE_SERVE_REGION, reversed_azureSpeech_emotion_map, azuresSpeech_emotion_map,reversed_gesture_emotion_map
import azure.cognitiveservices.speech as speechsdk
import numpy as np
from azure.cognitiveservices.speech import AudioDataStream, SpeechConfig, SpeechSynthesizer


try:
    import grpc
except ImportError:
    grpc = None

try:
    import keyboard
except ImportError:
    keyboard = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    from streaming_server import audio2face_pb2_grpc, audio2face_pb2
except ImportError:
    audio2face_pb2_grpc = None
    audio2face_pb2 = None


class SpeechController:
    def __init__(self, lang="en-US", role=None, devices="mic", filename=None, timeout="500", delay="500"):
        self.start_time = 0
        self.speech_key, self.service_region = AZURE_SPEECH_KEY, AZURE_SERVE_REGION
        self.lang = lang
        self.speech_config = SpeechConfig(subscription=self.speech_key, region=self.service_region,
                                          speech_recognition_language=self.lang)

        if role is None:
            self.role = "YunfengNeural" if "CN" in lang else "GuyNeural"
        else:
            self.role = role

        self.pool = ThreadPoolExecutor(1)

        self.speech_config.set_speech_synthesis_output_format(
            speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm)
        self.speech_config.set_property_by_name("endSilenceTimeoutMs", timeout)  # 0.5秒的静默时间
        self.speech_config.set_property_by_name("speechCompleteDelayMs", delay)  # 0.5秒的延迟时间

        if devices == "mic":
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self.is_file = False
        else:
            self.is_file = True
            if filename:
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=False,
                                                           filename=filename)
            else:
                audio_config = speechsdk.audio.AudioConfig(use_default_microphone=False,
                                                           filename="audio/prepare.wav")
        self.continuous_done = False
        self.recognize_result = ""
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
        self.synthesizer = SpeechSynthesizer(speech_config=self.speech_config, audio_config=None)
        self.connected_lambda = False
        self.last_synthesis_result = None

    def set_output_format(self, output_format=speechsdk.SpeechSynthesisOutputFormat.Riff48Khz16BitMonoPcm):
        self.speech_config.set_speech_synthesis_output_format(output_format)
        self.speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=None)

    def close(self):
        self.pool.shutdown()

    def __get_microphone(self):
        if pyaudio is None:
            raise RuntimeError("pyaudio is required to enumerate microphone devices.")
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            device_info = p.get_device_info_by_index(i)
            if device_info.get('maxInputChannels') > 0 and "Realtek HD" not in device_info["name"]:
                devices.append(device_info)
        return devices

    def create_ssml_text(self, emotion="Default", degree=1.0, content=""):
        safe_content = escape(content, quote=False)
        if emotion == "Default":
            body = safe_content
        else:
            body = f"""<mstts:express-as style="{emotion}" styledegree="{degree}">{safe_content}</mstts:express-as>"""

        template = f"""<speak xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" 
        xmlns:emo="http://www.w3.org/2009/10/emotionml" version="1.0" xml:lang="{self.lang}"> <voice 
        name="{self.lang}-{self.role}"><s /> {body} <s /></voice></speak>"""
        return template

    def __raise_if_synthesis_failed(self, result):
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            return

        details = result.cancellation_details
        reason = details.reason if details is not None else result.reason
        error_details = details.error_details if details is not None else ""
        raise RuntimeError(
            f"Azure speech synthesis failed: reason={reason}, error={error_details}"
        )

    def __wav_duration(self, filename):
        with wave.open(filename, "rb") as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()

    def __extract_emotion(self, emotion_dict: dict):
        max_strength = 0.0
        max_emotion = "Default"
        for emotion, strength in emotion_dict.items():
            if strength > max_strength:
                max_emotion = emotion
                max_strength = strength
        print(f"Speech emotion:{max_emotion}")
        if max_emotion in reversed_azureSpeech_emotion_map.keys():
            return reversed_azureSpeech_emotion_map.get(max_emotion), max_strength
        else:
            return "Default", 0.0

    def __resolve_speech_style(self, emotion):
        if isinstance(emotion, dict):
            return self.__extract_emotion(emotion)
        return emotion, 1.0

    def generate_audio_streaming(self, emotion, content):

        emotion, degree = self.__resolve_speech_style(emotion)

        text = self.create_ssml_text(emotion, degree, content)

        result = self.synthesizer.start_speaking_ssml_async(text).get()
        self.last_synthesis_result = result

        audio_data_stream = AudioDataStream(result)

        return emotion, audio_data_stream

    def __write_temp_file(self, audio_buffer, idx, sample_rate=48000):
        with wave.open(f"./audio/temp/audio_clip{idx}.wav", "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.setcomptype('NONE', "not compressed")
            wav_file.writeframes(audio_buffer)

    def synthesis(self, emotion, content, filename, is_streaming=False, audio2face=None, server=None,
                  gesture_generator=None,
                  answer_index=None):

        if not is_streaming:
            emotion_label, degree = self.__resolve_speech_style(emotion)
            text = self.create_ssml_text(emotion_label, degree, content)
            start_generating_time = time()
            result = self.synthesizer.speak_ssml_async(text).get()
            self.last_synthesis_result = result
            self.__raise_if_synthesis_failed(result)
            if isinstance(filename, os.PathLike):
                filename = os.fspath(filename)
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            AudioDataStream(result).save_to_wav_file(filename)
            return emotion, self.__wav_duration(filename), start_generating_time

        emotion_label, audio_data_stream = self.generate_audio_streaming(emotion, content)

        if not isinstance(emotion, dict):
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
                "neutral": 0.0  # 用于最后计算
            }  # 加权的结果
            if emotion != "Default":
                emotion_dict[azuresSpeech_emotion_map[emotion]] = 1.0
            del emotion_dict["neutral"]
            gesture_emotion_dict = {reversed_gesture_emotion_map[azuresSpeech_emotion_map[emotion_label]]: 1}

        else:
            emotion_dict = emotion.copy()
            gesture_emotion_dict = emotion.copy()

        start_generating_time = time()

        sample_rate = 48000
        buffer_size = 48000
        times = 0

        if is_streaming and os.path.exists("./audio/temp"):
            for temp_audio_file in os.listdir("./audio/temp"):
                os.remove(os.path.join("./audio/temp", temp_audio_file))

        with wave.open(filename, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.setcomptype('NONE', "not compressed")
            audio_buffer = bytes(buffer_size)

            if is_streaming:
                if grpc is None or audio2face_pb2_grpc is None or audio2face_pb2 is None:
                    raise RuntimeError("grpc and streaming_server protobuf modules are required for legacy streaming.")
                with grpc.insecure_channel("localhost:50051") as channel:
                    print("Channel created!")

                    stub = audio2face_pb2_grpc.Audio2FaceStub(channel)

                    def generate_gesture(gesture_generator, temp_counts, gesture_emotion_dict):
                        first_pose = None if temp_counts == 1 else f"./gesture/temp/audio_clip{temp_counts - 1}.bvh"
                        gesture_generator.generate(f"./audio/temp/audio_clip{temp_counts}.wav",
                                                   f"./gesture/temp/audio_clip{temp_counts}.bvh",
                                                   first_pose=first_pose,
                                                   emotion_dict=gesture_emotion_dict)

                    def convert(data):
                        sampled_data = np.frombuffer(data, dtype=np.int16) / 32768.0
                        return sampled_data.astype(np.float32)

                    def make_generator():

                        try:
                            sleep_between_chunks = 0.04
                            block_until_playback_is_finished = False
                            sample_rate = 48000

                            interval_time = 10.0
                            interval_counts = interval_time / (buffer_size / sample_rate / 2)
                            # interval_start = time()

                            start_marker = audio2face_pb2.PushAudioRequestStart(
                                samplerate=sample_rate,
                                instance_name="/World/audio2face/PlayerStreaming",
                                block_until_playback_is_finished=block_until_playback_is_finished,
                            )
                            yield audio2face_pb2.PushAudioStreamRequest(start_marker=start_marker)

                            nonlocal start_generating_time
                            filled_size = audio_data_stream.read_data(audio_buffer)

                            print(f"first block {time() - start_generating_time:.2f}")
                            wav_file.writeframes(audio_buffer)

                            temp_audio_buffer = bytearray(audio_buffer)

                            start_generating_time = time()

                            if audio2face is not None:
                                nonlocal emotion_dict
                                audio2face.set_livelink()
                                audio2face.set_emotions(emotion_dict)

                            yield audio2face_pb2.PushAudioStreamRequest(
                                audio_data=convert(audio_buffer).tobytes()
                            )

                            nonlocal times
                            times += 1
                            temp_counts = 1

                            while filled_size > 0:
                                sleep(sleep_between_chunks)
                                filled_size = audio_data_stream.read_data(audio_buffer)

                                if filled_size == 0:
                                    self.__write_temp_file(temp_audio_buffer, temp_counts)
                                    break

                                wav_file.writeframes(audio_buffer)
                                temp_audio_buffer = temp_audio_buffer + audio_buffer

                                # print(convert(audio_buffer).tobytes())
                                yield audio2face_pb2.PushAudioStreamRequest(
                                    audio_data=convert(audio_buffer).tobytes()
                                )

                                times += 1

                                if times % int(interval_counts) == 0 and gesture_generator is not None:
                                    self.__write_temp_file(temp_audio_buffer, temp_counts)
                                    # print(time() - interval_start)
                                    nonlocal gesture_emotion_dict
                                    self.pool.submit(generate_gesture, gesture_generator, temp_counts, gesture_emotion_dict)

                                    temp_audio_buffer = bytearray(0)
                                    temp_counts += 1

                        except Exception as e:
                            print(e)

                    request_generator = make_generator()
                    print("Sending audio data...")
                    response = stub.PushAudioStream(request_generator)
                    audio_length = buffer_size / sample_rate / 2 * times
                    if response.success:
                        print("Success!")
                    else:
                        print(f"ERROR: {response.message}")
            else:
                filled_size = audio_data_stream.read_data(audio_buffer)
                start_generating_time = time()
                wav_file.writeframes(audio_buffer)
                times += 1
                while filled_size > 0:
                    filled_size = audio_data_stream.read_data(audio_buffer)
                    if filled_size == 0:
                        break
                    wav_file.writeframes(audio_buffer)
                    times += 1
                audio_length = buffer_size / sample_rate / 2 * times

        return emotion, audio_length, start_generating_time

    def recognize(self, method="once"):

        recognize_time = 0
        start_time = time()

        self.recognize_result = ""
        self.continuous_done = False

        if method != "once":
            def on_recognized(evt):
                if self.recognize_result == "":
                    self.recognize_result += evt.result.text
                else:
                    self.recognize_result += " " + evt.result.text
                print('Recognized: {}'.format(evt))
                self.speech_recognizer.stop_continuous_recognition_async()
                nonlocal recognize_time
                nonlocal start_time
                recognize_time = time() - start_time
                self.continuous_done = True

            def stop(evt):
                self.speech_recognizer.stop_continuous_recognition_async()
                print('Stopped {}'.format(evt))

            if not self.connected_lambda:
                self.speech_recognizer.recognizing.connect((lambda evt: print('Recognizing: {}'.format(evt))))
                self.speech_recognizer.recognized.connect(on_recognized)
                self.speech_recognizer.session_started.connect(lambda evt: print('Session started: {}'.format(evt)))
                self.speech_recognizer.session_stopped.connect(stop)
                self.speech_recognizer.canceled.connect(stop)
                self.connected_lambda = True

            self.speech_recognizer.start_continuous_recognition_async()

            while not self.continuous_done:
                sleep(0.1)

            self.continuous_done = False
        else:
            res = self.speech_recognizer.recognize_once_async().get()
            self.recognize_result = res.text
            recognize_time = time() - start_time

        print("Recognized: {}".format(self.recognize_result))
        return self.recognize_result.strip(), recognize_time


if __name__ == "__main__":
    speech_controller = SpeechController("zh-CN")
    # print("Please pressing space!")
    # while True:
    #     if keyboard.is_pressed('space'):
    #         speech_controller.recognize()
    #     sleep(0.1)
    #

    # content = ("《活着》讲述了在大时代背景下，随着内战、三反五反、大跃进、"
    #            "“文化大革命”等社会变革，徐福贵的人生和家庭不断经受着苦难，"
    #            "到了最后所有亲人都先后离他而去，仅剩下年老的他和一头老牛相依为命。"
    #            "小说以普通、平实的故事情节讲述了在急剧变革的时代中福贵的不幸遭遇和坎坷命运，"
    #            "在冷静的笔触中展现了生命的意义和存在的价值，揭示了命运的无奈，与生活的不可捉摸。")

    while True:
        if keyboard.is_pressed("space"):
            content, _ = speech_controller.recognize()
            print(f"Recognized Time: {_:.2f}, including speech time!")
        sleep(0.1)

    # with open("./audio/test/file.txt", "r", encoding="utf-8") as f:
    #     content = f.read()
    #     print(content)

    # audio2face = Audio2FaceController(is_streaming=True)
    #
    # gesture_generator = GestureGenerator(f"{SPEECH2GESTURE_MODEL_PATH}/options.json", seed=int(time()),
    #                                      use_thread=False, use_gpu=True,
    #                                      use_preload=True,
    #                                      is_blocking=True, generating_mode=STREAMING_AND_FILE,
    #                                      load_size=1)
    #
    # gesture_generator.generating_mode = FILE
    # gesture_generator.generate(f"./audio/prepare.wav",
    #                            f"./gesture/prepare.bvh",
    #                            verbose_level=2)
    #
    # gesture_generator.generating_mode = STREAMING_AND_FILE
    # try:
    #     start_time = time()
    #     res, audio_length, playing_pos = speech_controller.synthesis(reversed_azureSpeech_emotion_map["neutral"],
    #                                                                  content,
    #                                                                  f"./audio/test.wav", is_streaming=True,
    #                                                                  audio2face=audio2face,
    #                                                                  )
    # except Exception as e:
    #     pass
    # finally:
    #     print(time() - start_time)
    #     print(res)
