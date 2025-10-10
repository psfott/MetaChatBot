from time import sleep

import keyboard
from pydub import AudioSegment, playback
from selenium import webdriver
from selenium.webdriver.common.by import By

from azure_speech import SpeechController
from global_data import reversed_azureSpeech_emotion_map, AUDIO_ROOT_PATH, background_emotion_map


def play(text):
    sentences = text.split("\n")
    if len(sentences) > 0:
        for sentence in sentences:
            if sentence not in played_sentence and "▌" not in sentence:
                print(sentence)
                speech_controller.synthesis(reversed_azureSpeech_emotion_map[background_emotion_map[condition]],
                                            sentence,
                                            f"{AUDIO_ROOT_PATH}/speech_only.wav")
                sound = AudioSegment.from_file(f"{AUDIO_ROOT_PATH}/speech_only.wav", format="wav")
                playback.play(sound)
                played_sentence.append(sentence)


# 先启动相关服务器
if __name__ == '__main__':
    browser = webdriver.Edge()
    browser.get('http://localhost:7860')
    speech_controller = SpeechController()

    response_index = 1

    condition = "happy"
    # condition = "neutral"
    # condition = "happy"
    # condition = "sad"

    selector_text = "#chatbot > div.wrapper.svelte-nab2ao > div > div > div:nth-child({child}) > div > button"

    while True:
        if keyboard.is_pressed("space"):
            prompt, recognize_time = speech_controller.recognize()
            if recognize_time != 0:
                text_area = browser.find_element(By.XPATH, "//textarea")
                text_area.send_keys(prompt)
                browser.find_element(By.ID, "component-13").click()
                sleep(1)
                content = browser.find_element(By.CSS_SELECTOR, selector_text.format(child=response_index * 2))
                length = len(content.text)
                sleep(0.1)
                played_sentence = []
                while length != len(content.text):
                    length = len(content.text)
                    sleep(0.3)
                    play(content.text)
                play(content.text)
                response_index = response_index + 1
