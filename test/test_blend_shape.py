import threading
from time import sleep

import librosa
import numpy as np

import json
import os.path

import subprocess
import socket
from pydub import AudioSegment
from pydub.playback import play

frame_header = {"Audio2Face": {"Facial": {}, "Body": {}, }}


def extract_video():
    video_dir = "./blendshape/raw_video"
    audio_dir = "./blendshape/audio"
    for file in os.listdir(video_dir):
        # ffmpeg 命令
        ffmpeg_command = [
            'ffmpeg',
            '-i', os.path.join(video_dir, file),  # 输入文件
            '-vn',  # 无视频
            '-ar', '48000',  # 设置音频采样率为48000Hz
            os.path.join(audio_dir, file.replace(".mp4", ".wav"))  # 输出文件
        ]

        # 调用 ffmpeg 并等待完成
        try:
            subprocess.run(ffmpeg_command, check=True)
            print(f'Audio extracted successfully to {audio_dir}')
        except subprocess.CalledProcessError as e:
            print(f'An error occurred while extracting audio: {e}')


def compare_remap_file():
    metahuman_blend_shape = []
    with open("./blendshape/remap/MetaHuman.txt", "r", encoding="utf-8-sig") as f:
        for line in f:
            metahuman_blend_shape.append(line.strip())
        print(metahuman_blend_shape)

    with open("./blendshape/json/a2f_export_bsweight_S1E17_332.json", "r") as f:
        audio2face_json = json.load(f)
        audio2face_blend_shape = audio2face_json["facsNames"]

    print(audio2face_blend_shape)


def calc_fps(input_path):
    audio_file_name = str(os.path.basename(input_path).replace("npy", "wav"))
    sound = AudioSegment.from_wav(os.path.join("./blendshape/audio", audio_file_name))
    audio_data, sample_rate = librosa.load(os.path.join("./blendshape/audio", audio_file_name))
    duration = librosa.get_duration(y=audio_data, sr=sample_rate)

    animation_length = np.load(input_path).shape[0]
    fps = animation_length / duration

    print(fps)
    return fps, sound


def send_frame(input_path):
    frame_data = np.load(input_path)

    metahuman_blend_shape = {}
    with open("./blendshape/remap/MetaHuman.txt", "r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            metahuman_blend_shape[line.strip()] = idx

    mediapipe_blend_shape = []
    with open("./blendshape/remap/mediapipe.txt", "r", encoding="utf-8") as f:
        for line in f:
            feature_name = line[0].capitalize() + line[1:].strip()
            mediapipe_blend_shape.append(feature_name)

    frame_json = frame_header.copy()
    facial_data = frame_json["Audio2Face"]["Facial"]
    facial_data["Names"] = list(metahuman_blend_shape.keys())
    facial_data["Weights"] = None
    total_weight = []

    buffer_size = 1024
    bytes_data = bytearray()

    facial_data["Weights"] = list(np.zeros(55, dtype=np.float64))
    json_str = json.dumps(frame_json)
    length = len(json_str)
    bytes_data = bytes_data + length.to_bytes(8, byteorder="big") + json_str.encode("utf-8")

    yield bytes_data  # 清零
    # sleep(1)

    neutral = 0
    for single_frame in frame_data:
        facial_data["Weights"] = list(np.zeros(55, dtype=np.float64))

        for idx, data in enumerate(single_frame):
            if idx >= 1:
                key_idx = metahuman_blend_shape[mediapipe_blend_shape[idx]]
                # if "eye" in mediapipe_blend_shape[idx].lower() or "brow" in mediapipe_blend_shape[idx].lower():
                #     data = 0.0
                facial_data["Weights"][key_idx] = max(0.0, data - neutral)
            else:
                neutral = data

        json_str = json.dumps(frame_json)
        length = len(json_str)
        bytes_data = bytes_data + length.to_bytes(8, byteorder="big") + json_str.encode("utf-8")

        while len(bytes_data) >= buffer_size:
            yield bytes_data
            print(bytes_data)
            bytes_data = bytearray()

        total_weight.append(facial_data["Weights"].copy())

    if len(bytes_data) > 0:
        yield bytes_data

    bytes_data = bytearray()
    yield bytes_data  # 清零

    facial_data["Weights"] = total_weight
    yield frame_json


if __name__ == "__main__":
    input_file = "./blendshape/bs/S1E1_392.npy"
    output_path = "./blendshape/json"
    # target 12030
    files = os.listdir("./blendshape/bs")

    files = [file for file in files if "117" in file]

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    hostname = '127.0.0.1'  # 示例使用本地地址
    port = 12030
    client_socket.connect((hostname, port))
    played = False

    while True:
        for file in files:
            file_path = os.path.join("./blendshape/bs", file)
            fps, sound = calc_fps(file_path)
            for frame in send_frame(file_path):
                if isinstance(frame, dict):
                    # output_name = str(os.path.basename(file_path).replace("npy", "json"))
                    # with open(os.path.join(output_path, output_name), "w") as f:
                    #     f.write(json.dumps(frame, indent=2))
                    pass
                else:
                    client_socket.sendall(frame)
                    if not played:
                        audio_thread = threading.Thread(target=play, args=(sound,))
                        audio_thread.start()
                        played = True
                    sleep(1 / fps)

            sleep(3)
            played = False
