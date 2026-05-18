# 医疗对话实验 TODO：Azure + Speech2Gesture + UE 5.6 ACE

## 目标

基于 `prompt/medical/script.json` 中的医生-患者对话脚本，生成一套可用于 UE 实验播放的素材：

```text
audio/medical/script1_001_doctor.wav
gesture/medical/script1_001_doctor.bvh
metadata/medical/script1_001_doctor.json
metadata/medical/script1_manifest.json
```

实验中使用 UE 5.6 ACE 插件根据同一段 wav 驱动口型和面部动画，使用 speech2gesture 根据同一段 wav 生成身体动作 BVH。UE 端以音频播放时间作为唯一同步时钟。

## 1. 实验条件

当前研究设计关注虚拟医生非语言行为对医疗共情感知的影响。实验变量应拆成两套独立控制：

```text
表情条件：无表情 / 有共情表情
动作条件：无身体动作 / 有共情身体动作
医生性别：男 / 女
```

形成医生性别 x 表情 x 动作的实验组合。

四种非语言条件：

```text
1. 无表情 + 无动作
2. 有表情 + 无动作
3. 无表情 + 有动作
4. 有表情 + 有动作
```

注意：实验操控对象建议主要放在 doctor 身上。patient 的情绪可以用于语音和剧情一致性，但不应把 patient 的表情/动作作为主要实验操控，否则会干扰“医生非语言行为”的自变量。

## 2. 情绪控制原则

不建议维护两套完全独立的 emotion。推荐结构是：

```text
一套语义情绪 semantic_emotion
        ↓
派生出两套实验控制
        ├─ face_control：控制 UE ACE 表情
        └─ gesture_control：控制 speech2gesture 身体动作
```

也就是说，医生这句话的情绪语义只有一份，但表情和动作是否启用、启用强度、使用哪种风格要分开控制。

统一 emotion 维度：

```text
neutral
joy
sadness
anger
fear
disgust
amazement
```

医生的共情状态建议主要使用：

```text
neutral + sadness + light joy
```

患者的情绪建议随剧情变化：

```text
前期：fear + sadness
中期：fear 降低，neutral 上升
后期：neutral + light joy
```

## 3. 脚本标准化

当前 `prompt/medical/script.json` 是这种形式：

```json
{"doctor": "Hello, please have a seat..."}
```

生成前应转换成标准内部结构：

```json
{
  "script_id": "script1",
  "index": 1,
  "speaker": "doctor",
  "text": "Hello, please have a seat...",
  "semantic_emotion": {
    "neutral": 0.7,
    "joy": 0.15,
    "sadness": 0.15,
    "anger": 0.0,
    "fear": 0.0,
    "disgust": 0.0,
    "amazement": 0.0
  },
  "face_control": {
    "enabled": true,
    "ace_override_strength": 0.2,
    "emotion": {
      "neutral": 0.7,
      "joy": 0.15,
      "sadness": 0.15
    }
  },
  "gesture_control": {
    "enabled": true,
    "emotion": {
      "neutral": 0.7,
      "joy": 0.15,
      "sadness": 0.15
    },
    "style": "empathic"
  }
}
```

第一版可以不直接改 `script.json`，而是在生成脚本中根据 speaker 和 index 自动补默认情绪。后续为了实验可复现，建议把每句的 `semantic_emotion` 显式写入脚本或单独写入标注文件。

## 4. 情绪映射

项目现有映射在 `global_data.py` 中已经有一部分，可以复用。

Azure Speech 映射：

```text
neutral   -> Default
joy       -> cheerful
sadness   -> sad
anger     -> angry
fear      -> terrified
disgust   -> unfriendly
amazement -> excited
```

speech2gesture / ZeroEGGS 映射：

```text
neutral   -> neutral
joy       -> happy
sadness   -> sad
anger     -> angry
fear      -> scared
disgust   -> sneaky
amazement -> laughing
```

ACE 表情映射：

```text
优先使用 semantic_emotion 中的情绪值
通过 ace_override_strength 控制外部情绪覆盖强度
不要把 override 拉满，避免表情过戏
```

推荐强度：

```text
doctor face_control.ace_override_strength: 0.15 - 0.30
patient face_control.ace_override_strength: 0.20 - 0.40
```

Azure 语音强度建议：

```text
azure_degree = clamp(max_emotion_strength, 0.2, 0.7)
```

医疗对话里语气要克制，尤其 doctor 不要过度 cheerful 或 sad。

## 5. 新增生成脚本

新增脚本：

```text
script_medical_play.py
```

职责：

```text
1. 读取 prompt/medical/script.json
2. 标准化每句对白
3. 为每句生成 Azure TTS wav
4. 为每句生成 speech2gesture BVH
5. 为每句写 metadata json
6. 写一个 script1_manifest.json，供 UE 顺序播放
```

输出目录：

```text
audio/medical/
gesture/medical/
metadata/medical/
```

文件命名：

```text
script1_001_doctor.wav
script1_001_doctor.bvh
script1_001_doctor.json
script1_002_patient.wav
script1_002_patient.bvh
script1_002_patient.json
```

## 6. Azure 语音生成

第一版只使用非流式：

```python
SpeechController.synthesis(
    emotion,
    text,
    wav_path,
    is_streaming=False
)
```

要求：

```text
1. 输出保持 48kHz mono wav，供 UE ACE 使用
2. 不使用旧 Omniverse Audio2Face gRPC
3. Azure key 和 region 后续改成环境变量
```

环境变量：

```text
AZURE_SPEECH_KEY
AZURE_SPEECH_REGION
```

医生和患者建议使用不同 voice：

```text
doctor_male
doctor_female
patient
```

具体 voice 后续根据 UE 中医生性别条件确定。

## 7. 身体动作生成

使用已经跑通的 speech2gesture 推理：

```python
gesture_generator.generate(
    wav_path,
    bvh_path,
    emotion_dict=gesture_control["emotion"],
    verbose_level=1
)
```

规则：

```text
gesture_control.enabled = true  -> 生成并播放共情动作 BVH
gesture_control.enabled = false -> 不播放动作，UE 使用 idle 或默认站姿
```

第一版每句独立生成即可。后续如果要减少句间身体跳变，再启用：

```python
first_pose=previous_bvh_path
```

## 8. UE 5.6 ACE 播放规则

UE 端必须以音频作为主时钟：

```text
同一个 wav
   ├─ ACE 插件驱动口型和脸部动画
   └─ 对应 BVH 驱动身体动作
```

播放一行对白时：

```text
1. 读取 metadata
2. 加载 wav
3. 如果 face_control.enabled = true，给 ACE 传 emotion override
4. 如果 face_control.enabled = false，ACE 只保留口型，关闭或弱化情绪表情
5. 如果 gesture_control.enabled = true，从 0 秒播放对应 BVH
6. 如果 gesture_control.enabled = false，保持 idle/默认动作
7. 口型、音频、身体动作都从同一个 audio playback time = 0 开始
```

不要让 Python 和 UE 分别启动计时器。Python 只负责预生成素材和 metadata。

## 9. Metadata 格式

每句 metadata 示例：

```json
{
  "script_id": "script1",
  "index": 1,
  "speaker": "doctor",
  "text": "Hello, please have a seat...",
  "audio": "audio/medical/script1_001_doctor.wav",
  "bvh": "gesture/medical/script1_001_doctor.bvh",
  "semantic_emotion": {
    "neutral": 0.7,
    "joy": 0.15,
    "sadness": 0.15,
    "anger": 0.0,
    "fear": 0.0,
    "disgust": 0.0,
    "amazement": 0.0
  },
  "azure": {
    "style": "Default",
    "degree": 0.7
  },
  "face_control": {
    "enabled": true,
    "ace_override_strength": 0.2,
    "emotion": {
      "neutral": 0.7,
      "joy": 0.15,
      "sadness": 0.15
    }
  },
  "gesture_control": {
    "enabled": true,
    "style": "empathic",
    "emotion": {
      "neutral": 0.7,
      "joy": 0.15,
      "sadness": 0.15
    }
  }
}
```

Manifest 示例：

```json
{
  "script_id": "script1",
  "items": [
    "metadata/medical/script1_001_doctor.json",
    "metadata/medical/script1_002_patient.json"
  ]
}
```

## 10. 验证清单

单句 doctor 验证：

```text
wav 成功生成
bvh 成功生成
metadata 成功生成
UE ACE 口型与 wav 对齐
医生表情不夸张
身体动作时长与音频接近
```

单句 patient 验证：

```text
wav 成功生成
如果 patient 不作为实验操控对象，可先不生成 bvh
语音情绪与剧情一致
```

四种实验条件验证：

```text
无表情 + 无动作：只有口型和 idle
有表情 + 无动作：ACE 有情绪表情，身体 idle
无表情 + 有动作：ACE 只有口型，身体有 gesture
有表情 + 有动作：ACE 表情和身体 gesture 同时启用
```

完整脚本验证：

```text
所有对白都有稳定编号
所有 wav/bvh/metadata 路径存在
UE 可以按 manifest 顺序播放
每句音频、口型、身体动作从同一时间启动
```

## 11. 后续改进

```text
1. 在 prompt/medical/script.json 中显式加入 semantic_emotion
2. 增加 doctor_male / doctor_female voice 配置
3. 增加实验条件生成器，自动导出 2 x 2 x 2 条件 manifest
4. 增加 first_pose 连接，减少医生动作跳变
5. 移除旧 Omniverse Audio2FaceController 依赖
6. 把 Azure key 从源码迁移到环境变量
7. 增加自动检查脚本，验证 wav/bvh/metadata 是否齐全
```
