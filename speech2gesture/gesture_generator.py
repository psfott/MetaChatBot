import json
import pathlib
import random
import threading
from pathlib import Path
from time import time, sleep

import numpy as np
import torch.jit
from omegaconf import DictConfig
from typing import Dict
import onnxruntime

from speech2gesture.anim import bvh
from speech2gesture.anim import quat
from speech2gesture.anim.txform import *
from speech2gesture.audio.audio_files import read_wavfile
from speech2gesture.data_pipeline import preprocess_animation
from speech2gesture.data_pipeline import preprocess_audio
from speech2gesture.utils import split_by_ratio, write_bvh
from global_data import reversed_gesture_emotion_map, gesture_emotion_map
from speech2gesture.bvh_loader import BVHLoader
import socket
from collections import deque
import global_data

global_data.init()


def symbolic_sinc(g, x):
    # 创建一个 ONNX 常量节点表示 pi
    pi_value = torch.as_tensor(torch.pi, device="cuda:0", dtype=torch.float32)
    pi_node = g.op("Constant", value_t=pi_value)

    # 创建一个 ONNX 节点表示 sin(pi*x)
    sin_input = g.op("Mul", pi_node, x)
    sin_output = g.op("Sin", sin_input)

    # 创建一个 ONNX 节点表示 pi*x
    mul_output = g.op("Mul", pi_node, x)

    # 创建一个 ONNX 节点表示 sinc(x) = sin(pi*x) / (pi*x)
    div_output = g.op("Div", sin_output, mul_output)

    return div_output


torch.onnx.register_custom_op_symbolic("aten::sinc", symbolic_sinc, 13)

FILE = 1
STREAMING = 2
STREAMING_AND_FILE = 3


class GestureGenerator:
    def __init__(self, option_file, temperature=random.random(), seed=time(), use_gpu=True, use_preload=True,
                 use_thread=True,
                 is_blocking=True, load_size=None, generating_mode=FILE):

        self.temperature = temperature
        self.use_preload = use_preload
        self.option_file = option_file
        self.seed = seed
        self.use_gpu = use_gpu
        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

        self.audio_features = None
        self.speech_encoding = None  # speech模型输入输出

        self.example_feature_vec = None
        self.final_style_encoding = None  # style模型输入输出

        self.network_decoder_input = None
        self.network_decoder_output = None  # decoder输入输出
        # 相关输入变量

        self.generating_mode = generating_mode

        if self.generating_mode != FILE:
            self.udpsocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udpsocket.bind(("localhost", 54322))
            self.target = ("localhost", 54321)
            self.frame_data_queue = deque()
            self.back_up_queue = deque()
            self.playing_thread = threading.Thread(target=self.sending_frame)
            self.running = True
            self.playing_thread.start()

        self.network_speech_encoder = None
        self.network_style_encoder = None
        self.network_decoder = None

        self.traced_network_decoder = None
        self.traced_network_speech_encoder = None
        self.traced_network_style_encoder = None
        # 追踪后模型
        with open(self.option_file, "r") as file:
            options = json.load(file)
        paths = options["paths"]
        self.base_path = Path(paths["base_path"])
        self.data_path = self.base_path / paths["path_processed_data"]
        self.network_path = Path(paths["models_dir"])
        self.bone_names = None
        self.dt = None
        self.path = None
        self.bvh_loader = BVHLoader(self.base_path)
        self.__imported = False
        self.load_size = load_size

        if use_preload:
            self.__load_model()
            self.bvh_loader.start(use_thread=use_thread, is_blocking=is_blocking, load_size=self.load_size)

        print("Load Gesture Generator Done!")

    def sending_frame(self):
        frame_count = 0
        restarting = False
        waiting_count = 60
        # back_count = 0
        while self.running:

            while len(self.frame_data_queue) == 0 and len(self.back_up_queue) != 0:

                while len(self.back_up_queue) > waiting_count:
                    self.back_up_queue.popleft()

                frame_data = self.back_up_queue.popleft()
                print(len(frame_data.encode()))
                self.udpsocket.sendto(frame_data.encode(), self.target)
                # back_count += 1
                restarting = True
                sleep(0.01)  # 倒放

            while len(self.frame_data_queue) > 0:
                if restarting:
                    self.back_up_queue.clear()
                    # print(f"Back count {back_count}")
                    back_count = 0
                    restarting = False

                frame_data = self.frame_data_queue.popleft()
                self.back_up_queue.append(frame_data)
                self.udpsocket.sendto(frame_data.encode(), self.target)
                frame_count += 1
                # print(f"Sending total frame {frame_count}")
                sleep(0.01)
            sleep(0.1)

    def cancel(self):
        self.running = False

    def __load_model(self):
        # 模型路径
        path_network_speech_encoder_weights = self.network_path / "speech_encoder.pt"
        path_network_decoder_weights = self.network_path / "decoder.pt"
        path_network_style_encoder_weights = self.network_path / "style_encoder.pt"
        path_stat_data = self.data_path / "stats.npz"
        path_data_definition = self.data_path / "data_definition.json"
        path_data_pipeline_conf = self.data_path / "data_pipeline_conf.json"
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(16)

        # Data pipeline conf (We must use the same processing configuration as the one in training)
        with open(path_data_pipeline_conf, "r") as f:
            data_pipeline_conf = json.load(f)
        self.data_pipeline_conf = DictConfig(data_pipeline_conf)
        # Animation static info (Skeleton, FPS, etc)
        with open(path_data_definition, "r") as f:
            details = json.load(f)
        self.bone_names = details["bone_names"]
        self.parents = torch.as_tensor(details["parents"], dtype=torch.long, device=self.device)
        self.dt = details["dt"]
        # 加载数据集统计数据
        stat_data = np.load(path_stat_data)
        self.audio_input_mean = torch.as_tensor(
            stat_data["audio_input_mean"], dtype=torch.float32, device=self.device
        )
        self.audio_input_std = torch.as_tensor(
            stat_data["audio_input_std"], dtype=torch.float32, device=self.device
        )
        self.anim_input_mean = torch.as_tensor(
            stat_data["anim_input_mean"], dtype=torch.float32, device=self.device
        )
        self.anim_input_std = torch.as_tensor(
            stat_data["anim_input_std"], dtype=torch.float32, device=self.device
        )
        self.anim_output_mean = torch.as_tensor(
            stat_data["anim_output_mean"], dtype=torch.float32, device=self.device
        )
        self.anim_output_std = torch.as_tensor(
            stat_data["anim_output_std"], dtype=torch.float32, device=self.device
        )
        # 加载模型
        self.network_speech_encoder = torch.load(str(path_network_speech_encoder_weights)).to(
            self.device)
        self.network_speech_encoder.eval()
        self.network_decoder = torch.load(path_network_decoder_weights).to(self.device)
        self.network_decoder.eval()
        self.network_style_encoder = torch.load(path_network_style_encoder_weights).to(
            self.device)
        self.network_style_encoder.eval()

    def export_model_and_animation(self):
        return {
            "bone_names": self.bone_names,
            "dt": self.dt,
            "parents": self.parents.cpu(),
            "data_pipeline_conf": self.data_pipeline_conf,
            "audio_input_mean": self.audio_input_mean.cpu(),
            "audio_input_std": self.audio_input_std.cpu(),
            "anim_input_mean": self.anim_input_mean.cpu(),
            "anim_input_std": self.anim_input_std.cpu(),
            "anim_output_mean": self.anim_output_mean.cpu(),
            "anim_output_std": self.anim_output_std.cpu(),
            "network_speech_encoder": self.network_speech_encoder.to("cpu"),
            "network_decoder": self.network_decoder.to("cpu"),
            "network_style_encoder": self.network_style_encoder.to("cpu"),
            "anim_data_load_info": self.bvh_loader.bvh_load_dict,
            "anim_data": self.bvh_loader.anim_data
        }

    def import_model_and_animation(self, data_dict):
        self.bone_names = data_dict["bone_names"]
        self.dt = data_dict["dt"]
        self.parents = data_dict["parents"].to(self.device)
        self.data_pipeline_conf = data_dict["data_pipeline_conf"]
        self.audio_input_mean = data_dict["audio_input_mean"].to(self.device)
        self.audio_input_std = data_dict["audio_input_std"].to(self.device)
        self.anim_input_mean = data_dict["anim_input_mean"].to(self.device)
        self.anim_input_std = data_dict["anim_input_std"].to(self.device)
        self.anim_output_mean = data_dict["anim_output_mean"].to(self.device)
        self.anim_output_std = data_dict["anim_output_std"].to(self.device)
        self.network_speech_encoder = data_dict["network_speech_encoder"].to(self.device)
        self.network_speech_encoder.eval()
        self.network_decoder = data_dict["network_decoder"].to(self.device)
        self.network_decoder.eval()
        self.network_style_encoder = data_dict["network_style_encoder"].to(self.device)
        self.network_style_encoder.eval()
        self.bvh_loader.bvh_load_dict = data_dict["anim_data_load_info"]
        self.bvh_loader.anim_data = data_dict["anim_data"]
        self.__imported = True

    def export_onnx(self, export_path):
        if self.audio_features is None or self.speech_encoding is None or \
                self.example_feature_vec is None or self.final_style_encoding is None \
                or self.network_decoder_input is None or self.network_decoder_output is None:
            print("You should call generate first!")
            return

        if self.traced_network_speech_encoder is None and self.traced_network_style_encoder is None and \
                self.traced_network_decoder is None:
            print("You should call trace model first!")
            return

        if self.traced_network_speech_encoder is not None:
            print("Exporting....")

            network_speech_encoder_input = (self.audio_features[
                                                np.newaxis] - self.audio_input_mean) / self.audio_input_std
            torch.onnx.export(self.traced_network_speech_encoder,
                              network_speech_encoder_input,
                              export_path + "speech_encoder.onnx",
                              export_params=True,
                              do_constant_folding=True,
                              input_names=["speech encoder input"],
                              output_names=["speech encoder pre-generate"],
                              verbose=True,
                              dynamic_axes={
                                  "speech encoder input": {1: 'audio feature'},
                                  "speech encoder pre-generate": {1: 'speech feature'}}
                              )
            print("Exporting speech encoder done!")

        if self.traced_network_style_encoder is not None:
            print("Exporting....")
            self.network_style_encoder.eval()

            network_style_encoder_input = (self.example_feature_vec[np.newaxis],
                                           torch.as_tensor([self.temperature], dtype=torch.float32,
                                                           device=self.device)
                                           )
            torch.onnx.export(self.traced_network_style_encoder,
                              network_style_encoder_input,
                              export_path + "style_encoder.onnx",
                              export_params=False,
                              do_constant_folding=True,
                              input_names=["style encoder input", "temperature"],
                              output_names=[" style_encoding", "temp1", "temp2"],
                              verbose=True
                              )
            print("Exporting style encoder done!")

        if self.traced_network_decoder is not None:
            decoder_input_name = ["root position", "root rotation", "root velocity", "root vrt", "lpos", "ltxy",
                                  "lvel",
                                  "lvrt",
                                  "gaze_pos", "speech encoding", "style encoding", "parents", "anim input mean",
                                  "anim input std",
                                  "anim pre-generate mean", "anim pre-generate", "dt"]

            decoder_output_name = ["root position", "root rotation", "root vel", "root vrt", "lpos", "ltxy", "vel",
                                   "vrt"]
            print("Exporting....")
            self.network_decoder.eval()

            network_decoder_input = list(self.network_decoder_input)
            network_decoder_input[-1] = torch.as_tensor([network_decoder_input[-1]], dtype=torch.float32,
                                                        device=self.device)
            torch.onnx.export(self.traced_network_decoder,
                              tuple(network_decoder_input),
                              export_path + "decoder.onnx",
                              export_params=True,
                              do_constant_folding=True,
                              input_names=decoder_input_name,
                              output_names=decoder_output_name,
                              verbose=True,
                              dynamic_axes={
                                  "gaze_pos": {1: 'audio feature'},
                                  "speech encoding": {1: 'audio feature'},
                                  "style encoding": {1: 'audio feature'},
                                  "root position": {1: 'audio feature'},
                                  "root rotation": {1: 'audio feature'},
                                  "root vel": {1: 'audio feature'},
                                  "root vrt": {1: 'audio feature'},
                                  "lpos": {1: 'audio feature'},
                                  "ltxy": {1: 'audio feature'},
                                  "vel": {1: 'audio feature'},
                                  "vrt": {1: 'audio feature'}
                              }
                              )
            print("Export decoder done!")

    def traced(self, model_name=None):

        if self.audio_features is None or self.speech_encoding is None or \
                self.example_feature_vec is None or self.final_style_encoding is None \
                or self.network_decoder_input is None or self.network_decoder_output is None:
            print("You should call generate first!")
            return

        if model_name is None:
            model_name = ["decoder"]

        if "speech" in model_name:
            self.network_speech_encoder.eval()
            network_speech_encoder_input = (self.audio_features[
                                                np.newaxis] - self.audio_input_mean) / self.audio_input_std
            self.traced_network_speech_encoder = torch.jit.trace(self.network_speech_encoder,
                                                                 network_speech_encoder_input)
            self.traced_network_speech_encoder.eval()
            self.traced_network_speech_encoder.to(self.device)
            print(self.traced_network_speech_encoder)

        if "decoder" in model_name:
            self.network_decoder.eval()
            network_decoder_input = list(self.network_decoder_input)
            network_decoder_input[-1] = torch.as_tensor([network_decoder_input[-1]], dtype=torch.float32,
                                                        device=self.device)
            self.traced_network_decoder = torch.jit.trace(self.network_decoder, tuple(network_decoder_input))
            self.traced_network_decoder.eval()
            self.traced_network_decoder.to(self.device)
            print(self.traced_network_decoder)

        if "style" in model_name:
            self.network_style_encoder.eval()
            network_style_encoder_input = (self.example_feature_vec[np.newaxis],
                                           torch.as_tensor([self.temperature], dtype=torch.float32,
                                                           device=self.device)
                                           )
            self.traced_network_style_encoder = torch.jit.script(self.network_style_encoder,
                                                                 network_style_encoder_input)
            self.traced_network_style_encoder.eval()
            self.traced_network_style_encoder.to(self.device)
            print(self.traced_network_style_encoder)

        return

    def __extract_style(self, emotion_dict: Dict[str, float]):
        max_strength = 0.0
        max_emotion = "neutral"
        styles = []
        if emotion_dict is None:
            return ["neutral"], 1.0
        for emotion, strength in emotion_dict.items():
            if strength > max_strength:
                max_emotion = emotion
                max_strength = strength
        print(f"Gesture emotion:{max_emotion}")
        if max_emotion in reversed_gesture_emotion_map.keys():
            styles.append(reversed_gesture_emotion_map.get(max_emotion))
            return styles, max_strength
        else:
            return ["neutral"], 0.0

    def generate(self, audio_file, file_name, encoding_file: str = None, emotion_dict: Dict[str, float] = None,
                 first_pose=None,
                 blend_type="add",
                 blend_ratio=None,
                 mode="",
                 audio_flag=None,
                 gesture_flag=None,
                 server=None,
                 verbose_level=0):
        total_time = time()

        if audio_flag is not None and gesture_flag is not None:
            answer_index = Path(audio_file).stem[-1]
            if str.isdigit(answer_index):
                answer_index = int(answer_index)

        if not self.use_preload and not self.__imported:
            print("Please Wait for loading models and animations!")
            self.__load_model()
            if encoding_file is None:
                self.bvh_loader.start(use_thread=False, is_blocking=True, load_size=self.load_size)
            self.use_preload = True

        if first_pose is not None:
            first_pose = Path(first_pose)

        style_tuples = None

        if blend_ratio is None:
            blend_ratio = [0.5, 0.5]

        if encoding_file != "" and encoding_file is not None:
            style_tuples = [(Path(encoding_file), None)]
        else:
            styles, _ = self.__extract_style(emotion_dict)
            fastload = True if encoding_file is not None else False
            style_tuples = [(self.bvh_loader.get_style_file(style, fastload), None) for style in styles]  # 随机抽取文件

        with ((torch.no_grad())):
            # If audio is None we only pre-generate the style encodings
            if audio_file is not None:
                # Load Audio
                if audio_flag is not None:
                    while not audio_flag[answer_index]:
                        sleep(0.1)
                # 等待声音合成

                start = time()
                _, audio_data = read_wavfile(
                    audio_file,
                    rescale=True,
                    desired_fs=16000,
                    desired_nb_channels=None,
                    out_type="float32",
                    logger=None,
                )
                if verbose_level == 2:
                    print(f"Read Time: {time() - start : .2f}")

                audio_length = len(audio_data) / 16000
                print(f"Audio length {audio_length:.2f}")

                n_frames = int(round(60.0 * audio_length))

                start = time()
                self.audio_features = torch.as_tensor(
                    preprocess_audio(
                        audio_data,
                        60,
                        n_frames,
                        self.data_pipeline_conf.audio_conf,
                        feature_type=self.data_pipeline_conf.audio_feature_type,
                    ),
                    device=self.device,
                    dtype=torch.float32,
                )
                if verbose_level == 2:
                    print(f"Extract feature Time: {time() - start : .2f}")

                if self.traced_network_speech_encoder is not None:
                    network_speech_encoder = self.network_speech_encoder if "traced" != mode else \
                        self.traced_network_speech_encoder
                else:
                    network_speech_encoder = self.network_speech_encoder
                network_speech_encoder.eval()

                start = time()
                self.speech_encoding = network_speech_encoder(
                    (self.audio_features[np.newaxis] - self.audio_input_mean) / self.audio_input_std
                )
                if verbose_level:
                    print(f"Speech encoding time {time() - start:.2f}.")

            # Style Encoding
            style_encodings = []

            for style in style_tuples:
                if isinstance(style[0], pathlib.WindowsPath) or isinstance(style[0], pathlib.PosixPath):

                    bvh_path = Path(style[0])

                    if encoding_file:
                        if str(bvh_path) not in self.bvh_loader.bvh_load_dict.keys():
                            self.bvh_loader.load_bvh(bvh_path)

                    while not self.bvh_loader.bvh_load_dict[str(bvh_path)]:
                        print(f"\rWaiting for loading animation {bvh_path}.")
                        sleep(5)

                    # Extracting features
                    (
                        root_pos,
                        root_rot,
                        root_vel,
                        root_vrt,
                        lpos,
                        lrot,
                        ltxy,
                        lvel,
                        lvrt,
                        cpos,
                        crot,
                        ctxy,
                        cvel,
                        cvrt,
                        gaze_pos,
                        gaze_dir,
                        rotations,
                    ) = self.bvh_loader.anim_data[str(bvh_path)]

                    # convert to tensor
                    start = time()
                    nframes = len(rotations)
                    root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=self.device)
                    root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=self.device)
                    root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=self.device)
                    root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=self.device)
                    lpos = torch.as_tensor(lpos, dtype=torch.float32, device=self.device)
                    ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=self.device)
                    lvel = torch.as_tensor(lvel, dtype=torch.float32, device=self.device)
                    lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=self.device)
                    gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=self.device)

                    S_root_vel = root_vel.reshape(nframes, -1)
                    S_root_vrt = root_vrt.reshape(nframes, -1)
                    S_lpos = lpos.reshape(nframes, -1)
                    S_ltxy = ltxy.reshape(nframes, -1)
                    S_lvel = lvel.reshape(nframes, -1)
                    S_lvrt = lvrt.reshape(nframes, -1)
                    example_feature_vec = torch.cat(
                        [
                            S_root_vel,
                            S_root_vrt,
                            S_lpos,
                            S_ltxy,
                            S_lvel,
                            S_lvrt,
                            torch.zeros_like(S_root_vel),
                        ],
                        dim=1,
                    )
                    if verbose_level == 2:
                        print(f"To tensor time {time() - start:.2f}.")

                    self.example_feature_vec = (example_feature_vec - self.anim_input_mean) / self.anim_input_std

                    if self.traced_network_style_encoder is not None:
                        network_style_encoder = self.network_style_encoder if "traced" != mode \
                            else self.traced_network_style_encoder
                    else:
                        network_style_encoder = self.network_style_encoder
                    network_style_encoder.eval()

                    start = time()
                    style_encoding, _, _ = network_style_encoder(
                        self.example_feature_vec[np.newaxis],
                        torch.as_tensor([self.temperature], dtype=torch.float32,
                                        device=self.device)
                    )
                    if verbose_level:
                        print(f"Style encoding time {time() - start:.2f}.")

                    style_encodings.append(style_encoding)

                elif isinstance(style[0], np.ndarray):
                    anim_name = style[1]
                    style_embeddding = torch.as_tensor(
                        style[0], dtype=torch.float32, device=self.device
                    )[np.newaxis]
                    style_encodings.append(style_embeddding)

            if blend_type == "stitch":
                if len(style_encodings) > 1:
                    if audio_file is None:
                        final_style_encoding = style_encodings
                    else:
                        assert len(styles) == len(blend_ratio)
                        se = split_by_ratio(n_frames, blend_ratio)
                        V_root_pos = []
                        V_root_rot = []
                        V_lpos = []
                        V_ltxy = []
                        final_style_encoding = []
                        for i, style_encoding in enumerate(style_encodings):
                            final_style_encoding.append(
                                style_encoding.unsqueeze(1).repeat((1, se[i][-1] - se[i][0], 1))
                            )
                        self.final_style_encoding = torch.cat(final_style_encoding, dim=1)
                else:
                    self.final_style_encoding = style_encodings[0]
            elif blend_type == "add":
                # style_encoding = torch.mean(torch.stack(style_encodings), dim=0)
                if len(style_encodings) > 1:
                    assert len(style_encodings) == len(blend_ratio)
                    self.final_style_encoding = torch.matmul(
                        torch.stack(style_encodings, dim=1).transpose(2, 1),
                        torch.tensor(blend_ratio, device=self.device),
                    )
                else:
                    self.final_style_encoding = style_encodings[0]

            if audio_file is not None:
                se = np.array_split(np.arange(n_frames), len(style_encodings))

                if first_pose is not None:
                    if isinstance(first_pose, pathlib.WindowsPath) or isinstance(first_pose,
                                                                                 pathlib.PosixPath):
                        if audio_flag is not None and gesture_flag is not None:
                            if answer_index != 1:
                                while not gesture_flag[answer_index - 1]:
                                    sleep(0.1)  # 等待上一个合成完毕
                        anim_data = self.bvh_loader.load_bvh(first_pose, self.bvh_loader.bvh_load_dict,
                                                             self.bvh_loader.anim_data, is_first_pose=True,
                                                             verbose=verbose_level)
                    elif isinstance(first_pose, dict):
                        anim_data = first_pose.copy()

                    start = time()
                    (
                        root_pos,
                        root_rot,
                        root_vel,
                        root_vrt,
                        lpos,
                        lrot,
                        ltxy,
                        lvel,
                        lvrt,
                        cpos,
                        crot,
                        ctxy,
                        cvel,
                        cvrt,
                        gaze_pos,
                        gaze_dir,
                        n_rotations,
                    ) = self.bvh_loader.anim_data[str(first_pose)]
                    if verbose_level == 2:
                        print(f"Look up time {time() - start:.2f}.")

                    root_vel = torch.as_tensor(root_vel, dtype=torch.float32, device=self.device)
                    root_vrt = torch.as_tensor(root_vrt, dtype=torch.float32, device=self.device)
                    root_pos = torch.as_tensor(root_pos, dtype=torch.float32, device=self.device)
                    root_rot = torch.as_tensor(root_rot, dtype=torch.float32, device=self.device)
                    lpos = torch.as_tensor(lpos, dtype=torch.float32, device=self.device)
                    ltxy = torch.as_tensor(ltxy, dtype=torch.float32, device=self.device)
                    lvel = torch.as_tensor(lvel, dtype=torch.float32, device=self.device)
                    lvrt = torch.as_tensor(lvrt, dtype=torch.float32, device=self.device)
                    gaze_pos = torch.as_tensor(gaze_pos, dtype=torch.float32, device=self.device)

                root_pos_0 = root_pos[0][np.newaxis]
                root_rot_0 = root_rot[0][np.newaxis]
                root_vel_0 = root_vel[0][np.newaxis]
                root_vrt_0 = root_vrt[0][np.newaxis]
                lpos_0 = lpos[0][np.newaxis]
                ltxy_0 = ltxy[0][np.newaxis]
                lvel_0 = lvel[0][np.newaxis]
                lvrt_0 = lvrt[0][np.newaxis]

                if self.final_style_encoding.dim() == 2:
                    self.final_style_encoding = self.final_style_encoding.unsqueeze(1).repeat(
                        (1, self.speech_encoding.shape[1], 1))

                gaze_pos_input = gaze_pos[0: 0 + 1].repeat_interleave(self.speech_encoding.shape[1],
                                                                      dim=0)[
                    np.newaxis
                ]

                self.network_decoder_input = (root_pos_0,
                                              root_rot_0,
                                              root_vel_0,
                                              root_vrt_0,
                                              lpos_0,
                                              ltxy_0,
                                              lvel_0,
                                              lvrt_0,
                                              gaze_pos_input,
                                              self.speech_encoding,
                                              self.final_style_encoding,
                                              self.parents,
                                              self.anim_input_mean,
                                              self.anim_input_std,
                                              self.anim_output_mean,
                                              self.anim_output_std,
                                              self.dt,
                                              )
                network_decoder = None
                if self.traced_network_decoder is not None:
                    network_decoder = self.network_decoder if "traced" != mode else self.traced_network_decoder
                else:
                    network_decoder = self.network_decoder
                network_decoder.eval()

                if verbose_level == 2:
                    network_decoder.set_verbose(True)
                else:
                    network_decoder.set_verbose(False)

                if self.generating_mode == FILE:
                    network_decoder.close_streaming_mode()

                    generator = network_decoder(
                        root_pos_0,
                        root_rot_0,
                        root_vel_0,
                        root_vrt_0,
                        lpos_0,
                        ltxy_0,
                        lvel_0,
                        lvrt_0,
                        gaze_pos_input,
                        self.speech_encoding,
                        self.final_style_encoding,
                        self.parents,
                        self.anim_input_mean,
                        self.anim_input_std,
                        self.anim_output_mean,
                        self.anim_output_std,
                        torch.as_tensor([self.dt], dtype=torch.float32, device=self.device),
                    )

                    if verbose_level:
                        print(f"Decoding time {time() - start:.2f}.")

                    for output in generator:
                        self.network_decoder_output = output
                        (
                            V_root_pos,
                            V_root_rot,
                            V_root_vel,
                            V_root_vrt,
                            V_lpos,
                            V_ltxy,
                            V_lvel,
                            V_lvrt,
                        ) = self.network_decoder_output

                else:
                    network_decoder.open_streaming_mode(self.bone_names, self.parents.detach().cpu().numpy(),
                                                        )
                    frame_datas = network_decoder(
                        root_pos_0,
                        root_rot_0,
                        root_vel_0,
                        root_vrt_0,
                        lpos_0,
                        ltxy_0,
                        lvel_0,
                        lvrt_0,
                        gaze_pos_input,
                        self.speech_encoding,
                        self.final_style_encoding,
                        self.parents,
                        self.anim_input_mean,
                        self.anim_input_std,
                        self.anim_output_mean,
                        self.anim_output_std,
                        torch.as_tensor([self.dt], dtype=torch.float32, device=self.device),
                    )

                    # total_list = {"test": []}

                    is_starting = False
                    for frame_data in frame_datas:
                        if server is not None and not is_starting:
                            server.send_data(f"Gesture pos:{1}")
                            is_starting = True

                        if not isinstance(frame_data, str):
                            self.network_decoder_output = frame_data
                            (
                                V_root_pos,
                                V_root_rot,
                                V_root_vel,
                                V_root_vrt,
                                V_lpos,
                                V_ltxy,
                                V_lvel,
                                V_lvrt,
                            ) = self.network_decoder_output
                        else:
                            # total_list["test"].append(json.loads(frame_data))
                            self.frame_data_queue.append(frame_data)  # 放入队列

                    # with open("../test/json/bvh_test.json", "w") as f:
                    #     f.write(json.dumps(total_list, indent=2))

                if self.generating_mode != STREAMING:
                    V_lrot = quat.from_xform(xform_orthogonalize_from_xy(V_ltxy).detach().cpu().numpy())

                    if file_name is None:
                        file_name = f"audio_{audio_file.stem}_label_{anim_name}"
                    try:
                        wirte_root_pos = V_root_pos[0].detach().cpu().numpy()
                        wirte_root_rot = V_root_rot[0].detach().cpu().numpy()
                        wirte_lpos = V_lpos[0].detach().cpu().numpy()
                        write_lrot = V_lrot[0]

                        start = time()
                        write_bvh(
                            file_name,
                            wirte_root_pos,
                            wirte_root_rot,
                            wirte_lpos,
                            write_lrot,
                            parents=self.parents.detach().cpu().numpy(),
                            names=self.bone_names,
                            order="zyx",
                            dt=self.dt,
                            start_position=np.array([0, 0, 0]),
                            start_rotation=np.array([1, 0, 0, 0]),
                        )
                        if verbose_level == 2:
                            print(f"Write bvh time {time() - start:.2f}.")

                        print(f"Generate animation at {file_name}")

                    except (PermissionError, OSError) as e:
                        print(e)

        torch.cuda.empty_cache()  # 一定要注意释放显存

        if verbose_level:
            print(f"Total Generation time {time() - total_time:.2f}.")
        return audio_length, self.final_style_encoding


if __name__ == "__main__":
    condition = "-break"
    gesture_generator = GestureGenerator("../model/full_data_options.json", seed=int(time()), use_thread=False,
                                         use_gpu=True,
                                         use_preload=True,
                                         is_blocking=True, generating_mode=STREAMING_AND_FILE,
                                         load_size=1)
    label = "happy"
    file_name =   f"../gesture/happy.bvh"
    index = 4
    emotion_dict = {gesture_emotion_map[label]: 1}
    audio_file = (r"F:\Audio2Face\emoitonal voice2\audio\emotion2-0002.wav")
    # print(emotion_dict)
    gesture_generator.generating_mode = STREAMING_AND_FILE
    gesture_generator.generate(audio_file,
                               file_name,
                               emotion_dict=emotion_dict,
                               verbose_level=2)

    # gesture_generator.generating_mode = STREAMING_AND_FILE
    # gesture_generator.generate(f"../audio/prepare.wav",
    #                            f"../gesture/test.bvh",
    #                            emotion_dict=emotion_dict,
    #                            verbose_level=2)
    #
    # gesture_generator.cancel()
    # gesture_generator.traced(["style", "speech"])
    # gesture_generator.export_onnx("../model/onnx/")
    # gesture_generator.generate("../audio/answer1.wav", "../gesture/test1.bvh",
    #                            first_pose="../gesture/test.bvh",
    #                            emotion_dict={"joy": 0.8}, verbose_level=2)
    # gesture_generator.generate("../audio/answer2.wav", "../gesture/test2.bvh",
    #                            first_pose="../gesture/test1.bvh",
    #                            emotion_dict={"joy": 0.8}, verbose_level=1)
    # gesture_generator.generate("../audio/answer3.wav", "../gesture/test3.bvh",
    #                            first_pose="../gesture/test2.bvh",
    #                            emotion_dict={"sadness": 0.8}, verbose_level=1)
    # global_data.get("process_pool").join()
