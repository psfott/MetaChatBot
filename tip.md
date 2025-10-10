# 注意

需要先启用audio2face headless模式，powershell命令：

```shell
./audio2face_headless.bat
cd D:\Omniverse\pkg\deps\3757f0f707549e4cd2e0c789de84c010
```

注意运行程序时显存占用，释放显存

```shell
./audio2face_headless.bat
cd C:\Users\admin\AppData\Local\ov\pkg\audio2face-2023.1.1
```

注意live link链接是否成功

# 数据集

需要有processed_data.npz文件，大约6GB。具体为ZEGGS数据库处理得到
<https://github.com/ubisoft/ubisoft-laforge-ZeroEGGS>

# FastChat测试

api版本

```shell
python -m fastchat.serve.controller
python -m fastchat.serve.model_worker --model-path lmsys/vicuna-7b-v1.5
python -m fastchat.serve.openai_api_server --host localhost --port 8088
python -m fastchat.serve.gradio_web_server
```

命令行版本

```shell
python -m fastchat.serve.cli --model-path lmsys/vicuna-7b-v1.5
```

# prompt

1. Suppose you are talking to a friend who saw someone thrown up in the backseat.
   Please talk with him about his experience and label
   one of the seven emotion tag (neutral, sadness, joy, anger, fear, disgust, amazement) at the end of each sentence.
   Here is an example of your response:
   "I'm sorry to hear that.[sadness] How can I help you?[neutral]"


2. In the following conversation, please analyze the possible emotion
   and intensity for your response to the User and return
   them to me in the following json format.
   {response":"your response", "amazement": 0.0, "anger": 0.0,
   "cheekiness": 0.0, "disgust": 0.0, "fear": 0.0,"grief": 0.0,
   "joy": 0.0, "outofbreath": 0.0, "pain": 0.0, "sadness": 0.0}User:"Can you help me?"

# Scene

见background

# Whipercpp
```shell
ffmpeg -i './audio/test.wav' -ar 16000 -ac 1 -c:a pcm_s16le "./audio/test/test_16Khz.wav"
```
```shell
./whispercpp/main.exe -m "F:\Model\hub\whisper\ggml-small.bin" -f "./audio/capture/recording.wav" -l auto --prompt "以下是普通话的句子"
```

## Sever
```shell
./whispercpp/server.exe -m "F:\Model\hub\whisper\ggml-small.bin"  -l auto --prompt "以下是普通话的句子" --port 8020
```
