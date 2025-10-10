import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
from time import time, sleep

import numpy as np
from tqdm import tqdm

from global_data import gesture_emotion_map
from speech2gesture.anim import bvh
from speech2gesture.data_pipeline import preprocess_animation
from multiprocessing import Pool, Manager, cpu_count
import global_data

global_data.init()


# 进程池全局变量

class BVHLoader:
    def __init__(self, base_dir):

        self.load_size = None
        self.base_path = Path(base_dir)
        self.bvh_list = [file for file in os.listdir(self.base_path / "clean") if
                         ".bvh" in file and "_" in file and file.split("_")[1].lower() in gesture_emotion_map.keys()]

        self.bvh_load_dict = {str(self.base_path / "clean" / file): False for file in self.bvh_list}

        self.anim_data = {}
        self.ban_list = []

        self.__get_ban_list()

    def __get_ban_list(self):
        base_path = os.getcwd()
        banlist_file_path = "../gesture/banlist.txt" if "speech2gesture" in base_path else "./gesture/banlist.txt"
        with open(banlist_file_path, "r", encoding="utf8") as f:
            for line in f:
                if "#" not in line:
                    self.ban_list.append(line.strip())

    def load_bvh(self, bvh_file, bvh_load_dict=None, anim_data_dict=None, is_first_pose=False, verbose=False):
        start = time()
        anim_data = bvh.load(bvh_file)
        if verbose:
            print(f"BVH {bvh_file} load time {time() - start:.2f}.")
        anim_fps = int(np.ceil(1 / anim_data["frametime"]))
        assert anim_fps == 60
        # 保存
        start = time()
        if is_first_pose:
            self.__process_first_pose(anim_data)
        data_list = list(preprocess_animation(anim_data))
        if verbose:
            print(f"BVH {bvh_file} process time {time() - start:.2f}.")
        data_list.append(anim_data["rotations"])
        if anim_data_dict is None:
            self.anim_data[str(bvh_file)] = tuple(data_list)
        else:
            anim_data_dict[str(bvh_file)] = tuple(data_list)

        if bvh_load_dict is None:
            self.bvh_load_dict[str(bvh_file)] = True
        else:
            bvh_load_dict[str(bvh_file)] = True

    def __process_first_pose(self, anim_data):
        mask = np.isnan(anim_data["rotations"][:, 0, 0])
        indexes = np.where(~mask)
        anim_data["rotations"] = anim_data["rotations"][indexes[0]]

        mask = np.isnan(anim_data["positions"][:, 0, 0])
        indexes = np.where(~mask)
        anim_data["positions"] = anim_data["positions"][indexes[0]]

        if anim_data["rotations"].shape[0] > anim_data["positions"].shape[0]:
            anim_data["rotations"] = anim_data["rotations"][0:len(anim_data["positions"])]
        elif anim_data["rotations"].shape[0] < anim_data["positions"].shape[0]:
            anim_data["positions"] = anim_data["positions"][0:len(anim_data["rotations"])]
        else:
            pass

        anim_data["rotations"] = anim_data["rotations"][
                                 -1:-11:-1,
                                 ].copy()
        anim_data["positions"] = anim_data["positions"][
                                 -1:-11:-1,
                                 ].copy()

    def worker(self, file_list, bvh_load_dict=None, anim_data_dict=None):
        bar = tqdm(range(len(file_list)), desc=f'Loading Animation...')
        for i in bar:
            bvh_file = file_list[i]

            if bvh_file not in self.ban_list:
                bvh_file = self.base_path / "clean" / bvh_file
                self.load_bvh(bvh_file, bvh_load_dict, anim_data_dict)
                bar.set_postfix_str(f"Load Animation {bvh_file} done!")

            sleep(0.01)

    def get_style_file(self, style, fastload=False):
        random.seed(int(time()))

        style_file_list = [style_file for style_file in self.bvh_list.copy() if style in style_file.lower()
                           and style not in self.ban_list]

        if len(style_file_list) != 0:
            filename = self.base_path / "clean" / random.choice(style_file_list)

            condition = True
            while condition:
                filename = self.base_path / "clean" / random.choice(style_file_list)
                condition = not self.bvh_load_dict[str(filename)] if isinstance(self.bvh_load_dict, dict) \
                    else str(filename) not in self.bvh_load_dict.keys()
                if fastload:
                    self.load_bvh(filename)
                    break
                if condition:
                    print(f"\rWaiting for loading animation {filename}")
                    sleep(5)

            print(f"Choose file:{filename}")
            return filename
        else:
            return NotImplementedError("No such Style!")

    def start(self, num_worker=int(cpu_count() / 3), use_thread=True, is_blocking=True, load_size=None):
        start = time()
        self.load_size = load_size

        # if load_size is not None:
        #     if load_size > 4:
        #         raise ValueError(f"Load size{load_size} is too large!")
        #     output = []
        #     for emotion in gesture_emotion_map.keys():
        #         print(self.bvh_list, self.ban_list, load_size)
        #         output.extend(random.sample(
        #             [file for file in self.bvh_list if emotion in file.lower() and file not in self.ban_list],
        #             k=load_size))
        #     self.bvh_list = output.copy()

        length = len(self.bvh_list)
        step = int(length / num_worker) + 1
        lists = [self.bvh_list[i:i + step] for i in range(0, length, step)]

        if use_thread:
            thread_pool = ThreadPoolExecutor(num_worker)
            for worker_list in lists:
                thread_pool.submit(self.worker, worker_list, None, None)
            if is_blocking:
                thread_pool.shutdown()
        else:
            manager = Manager()
            self.bvh_load_dict = manager.dict()
            self.anim_data = manager.dict()

            process_pool = Pool(num_worker)
            global_data.set("process_pool", process_pool)
            for worker_list in lists:
                process_pool.apply_async(func=self.worker, args=(worker_list, self.bvh_load_dict, self.anim_data))
            process_pool.close()
            if is_blocking:
                process_pool.join()

        if is_blocking:
            print(f"Finished at {time() - start:.2f}")


if __name__ == "__main__":
    bvh_loader = BVHLoader("F:/Github Repo/ubisoft-laforge-ZeroEGGS/data")
    start_time = time()
    bvh_loader.load_bvh("F:/Github Repo/ubisoft-laforge-ZeroEGGS/data/clean/001_Neutral_0_mirror_x_1_0.bvh",
                        verbose=True)
    print(f"Finished at {time() - start_time:.2f}")
    start_time = time()
    bvh_loader.load_bvh("F:/Github Repo/ubisoft-laforge-ZeroEGGS/data/clean/001_Neutral_0_mirror_x_1_0.bvh",
                        verbose=True)
    print(f"Finished at {time() - start_time:.2f}")
