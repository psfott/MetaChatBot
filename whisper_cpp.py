import subprocess
import wave
from time import time, sleep

import keyboard
import pyaudio
import requests
import webrtcvad


class WhisperCpp:
    def __init__(self, model="F:/Model/hub/whisper/ggml-small.bin", port=8020, lang="en-US"):
        self.audio = pyaudio.PyAudio()
        self.capture_path = "./audio/capture/recording.wav"
        self.port = port
        prompt = "以下是普通话的句子" if lang == "zh-CN" else ""
        self.cmds = ["./whispercpp/server.exe", "-m", model, "-l", "auto", "--prompt", prompt,
                     "--port", str(self.port)]
        self.vad = webrtcvad.Vad(1)

        self.url = f"http://127.0.0.1:{self.port}/inference"

        # 构造请求数据
        self.header_data = {
            'temperature': '0.0',
            'temperature_inc': '0.2',
            'response_format': 'json'
        }

        self.__run_server()

    def __run_server(self):
        self.process = subprocess.Popen(self.cmds,
                                        start_new_session=True,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        bufsize=1024,
                                        text=True)

        for line in self.process.stderr:
            print(line.strip())
            if "compute buffer (decode)" in line:
                print(f"Sever launched at {self.port}")
                break

    def close(self):
        self.audio.terminate()
        self.process.terminate()

    def capture(self, silence_duration=1):
        audio_format = pyaudio.paInt16  # 音频流的格式
        channels = 1  # 单通道
        rate = 16000  # 采样率16kHz

        frame_duration = 30
        chunk = int(rate * frame_duration / 1000)  # 每次读取的数据块大小

        # 打开音频流
        stream = self.audio.open(format=audio_format,
                                 channels=channels,
                                 rate=rate,
                                 input=True,
                                 frames_per_buffer=1024)

        # 记录音频数据
        print("Starting Recording...")
        frames = []

        stream.start_stream()

        speech_idx = 1
        start_time = time()

        try:
            while True:  # 阈值可以根据需要调整=
                data = stream.read(chunk)

                is_speech = self.vad.is_speech(data, rate)

                if is_speech:
                    frames.append(data)
                    start_time = time()
                    speech_idx = speech_idx + 1

                if time() - start_time > silence_duration:
                    print("Silence detected!")
                    break

                # audio_data = np.frombuffer(data, dtype=np.int16)
                # volume = np.linalg.norm(audio_data) / np.sqrt(len(audio_data))
                #
                # if volume < threshold:
                #     silent_chunks += 1
                # else:
                #     silent_chunks = 0

                # if silent_chunks > (silence_duration * rate / chunk):
                #     print("Silence detected, stopping recording.")
                #     break

        except Exception as e:
            print(f"Exception: {e}")

        # 停止并关闭音频流
        print("Stopping Recording...")
        stream.stop_stream()
        stream.close()

        total_frames = b''.join(frames)

        # 保存音频文件
        wf = wave.open(self.capture_path, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(self.audio.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(total_frames)
        wf.close()

        print(f"Saved {self.capture_path}")

    def recognize(self, file_path=None):

        if file_path is str:
            with open(file_path, "rb") as f:
                audio_data = (file_path, f.read())
        else:
            self.capture()
            with open(self.capture_path, "rb") as f:
                audio_data = (self.capture_path, f.read())
        # 设置URL和文件路径

        start = time()
        response = requests.post(self.url, files={'file': audio_data}, data=self.header_data).json()
        # 输出响应
        recognize_time = time() - start
        text = response["text"]

        return text, recognize_time


if __name__ == "__main__":
    whispercpp = WhisperCpp()

    while True:
        if keyboard.is_pressed('space'):
            res, recognize_time = whispercpp.recognize()
            print(f"Recognized: {res.strip()} {recognize_time:.2f}s!")
        sleep(0.1)
