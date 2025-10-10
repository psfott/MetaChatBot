import os.path

import requests

from global_data import AUDIO_ROOT_PATH, BASE_PATH


class Audio2FaceController:
    def __init__(self, root_path=BASE_PATH, is_streaming=False):
        self.base_url = "http://127.0.0.1:8011"
        if not is_streaming:
            self.player = "/World/audio2face/Player"
            self.scene_path = os.path.join(root_path, "Stage", "Stage.usd")
        else:
            self.player = "/World/audio2face/PlayerStreaming"
            self.scene_path = os.path.join(root_path, "Stage", "Stage_Streaming.usd")
        self.fullface_instance = "/World/audio2face/CoreFullface"
        self.node_path = "/World/audio2face/StreamLivelink"
        self.session = requests.Session()  # 获取session
        self.__init_audio2face_headless(root_path, is_streaming)

    def __init_audio2face_headless(self, root_path=BASE_PATH, is_streaming=False):
        print("Wait for scene loading......")
        self.load_scene()
        print("Scene loading done......")
        if not is_streaming:
            self.set_root_path(root_path)
            self.set_looping()
        self.set_livelink()

    def __post(self, url, data):
        headers = {"accept": "application/json"}
        response = self.session.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, response.json()

    def close(self):
        self.session.close()

    def set_root_path(self, root_path: str = AUDIO_ROOT_PATH):
        data = {"a2f_player": self.player, "dir_path": root_path}
        status, response = self.__post(self.base_url + "/A2F/Player/SetRootPath", data)
        print(response)

    def load_scene(self):
        data = {"file_name": self.scene_path}
        status, response = self.__post(self.base_url + "/A2F/USD/Load", data)
        print(response)

    def set_track(self, track_path: str):
        data = {"a2f_player": self.player, "file_name": track_path, "time_range": [0, -1]}
        status, response = self.__post(self.base_url + "/A2F/Player/SetTrack", data)
        print(response)

    def clear_emotion(self):
        data = {"a2f_instance": self.fullface_instance, "emotions": {
            "amazement": 0.0,
            "anger": 0.0,
            "cheekiness": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "grief": 0.0,
            "joy": 0.0,
            "outofbreath": 0.0,
            "pain": 0.0,
            "sadness": 0.0
        }}
        status, response = self.__post(self.base_url + "/A2F/A2E/SetEmotionByName", data)
        return status

    def set_emotion_by_name(self, emotion: str, strength: float):
        if self.clear_emotion():
            data = {"a2f_instance": self.fullface_instance, "emotions": {emotion: strength}}
            status, response = self.__post(self.base_url + "/A2F/A2E/SetEmotionByName", data)
            print(response)

    def set_emotions(self, emotions: dict):
        if "neutral" in emotions.keys():
            del emotions["neutral"]
        if self.clear_emotion() and emotions:
            data = {"a2f_instance": self.fullface_instance, "emotions": emotions}
            status, response = self.__post(self.base_url + "/A2F/A2E/SetEmotionByName", data)
            print(response)

    def set_looping(self, is_looping=False):
        data = {"a2f_player": self.player, "loop_audio": is_looping}
        status, response = self.__post(self.base_url + "/A2F/Player/SetLooping", data)
        print(response)

    def set_livelink(self, value=True):
        data = {"node_path": self.node_path, "value": value}
        status, response = self.__post(self.base_url + "/A2F/Exporter/ActivateStreamLivelink", data)
        print(response)

    def play(self):
        data = {"a2f_player": self.player}
        status, response = self.__post(self.base_url + "/A2F/Player/Play", data)
        print(response)


if __name__ == "__main__":
    audio2face = Audio2FaceController(is_streaming=False)
    audio2face.set_track("./audio/test.wav")
    audio2face.set_livelink(True)
    audio2face.play()
