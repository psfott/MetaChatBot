import argparse
import json
from pathlib import Path

from global_data import reversed_azureSpeech_emotion_map, reversed_gesture_emotion_map

EMOTION_ORDER = (
    "neutral",
    "amazement",
    "anger",
    "cheekiness",
    "disgust",
    "fear",
    "grief",
    "joy",
    "out_of_breath",
    "pain",
    "sadness",
)

ACE_EMOTION_KEYS = tuple(name for name in EMOTION_ORDER if name != "neutral")
GESTURE_EMOTION_KEYS = tuple(reversed_gesture_emotion_map.keys())

CONDITIONS = {
    "baseline": {"face": False, "gesture": False},
    "face": {"face": True, "gesture": False},
    "gesture": {"face": False, "gesture": True},
    "full": {"face": True, "gesture": True},
}


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def empty_emotion():
    return {name: 0.0 for name in EMOTION_ORDER}


def normalize_emotion(values):
    emotion = empty_emotion()
    emotion.update({key: float(value) for key, value in values.items() if key in emotion})
    total = sum(max(value, 0.0) for value in emotion.values())
    if total <= 0:
        emotion["neutral"] = 1.0
        return emotion
    return {key: max(value, 0.0) / total for key, value in emotion.items()}


def dominant_emotion(emotion):
    return max(emotion.items(), key=lambda item: item[1])


def default_doctor_emotion(text):
    lower = text.lower()
    if any(token in lower for token in ("good news", "reassuring", "relief", "helps a lot")):
        return {
            **empty_emotion(),
            "neutral": 0.08,
            "joy": 0.7,
            "sadness": 0.12,
            "fear": 0.04,
            "grief": 0.02,
        }
    if any(token in lower for token in ("risk", "complication", "progress", "pain", "infection", "urgent")):
        return {
            **empty_emotion(),
            "neutral": 0.08,
            "sadness": 0.62,
            "fear": 0.18,
            "grief": 0.08,
            "pain": 0.05,
            "joy": 0.12,
        }
    return {
        **empty_emotion(),
        "neutral": 0.08,
        "sadness": 0.5,
        "joy": 0.28,
        "fear": 0.08,
        "grief": 0.04,
    }


def default_semantic_emotion(speaker, text, index, total_items):
    if speaker == "doctor":
        return default_doctor_emotion(text)
    return neutral_emotion()


def classifier_semantic_emotion(emotion_classifier, text):
    raw_emotion = emotion_classifier.plain_classify(text)
    return ace_empathy_emotion(normalize_emotion(raw_emotion))


def neutral_emotion():
    emotion = empty_emotion()
    emotion["neutral"] = 1.0
    return emotion


def ace_empathy_emotion(emotion):
    output = empty_emotion()
    for key in output:
        output[key] = float(emotion.get(key, 0.0))
    return output


def condition_semantic_emotion(condition, base_emotion):
    if not CONDITIONS[condition]["face"]:
        return neutral_emotion()
    return ace_empathy_emotion(base_emotion)


def gesture_emotion_from_semantic(base_emotion):
    gesture_emotion = {
        key: float(base_emotion.get(key, 0.0))
        for key in GESTURE_EMOTION_KEYS
    }
    total = sum(max(value, 0.0) for value in gesture_emotion.values())
    if total <= 0:
        return {"neutral": 1.0, **{key: 0.0 for key in GESTURE_EMOTION_KEYS if key != "neutral"}}
    return {
        key: max(value, 0.0) / total
        for key, value in gesture_emotion.items()
    }


def semantic_emotion_for_turn(emotion_classifier, speaker, text, index, total_items):
    if emotion_classifier is None:
        return default_semantic_emotion(speaker, text, index, total_items), "heuristic"
    return classifier_semantic_emotion(emotion_classifier, text), "emotion_classifier"


def build_emotion_classifier(args):
    if not args.use_classifier:
        return None

    from emotion_classifier import EmotionClassifier

    return EmotionClassifier(args.classifier_lang)


def azure_metadata(emotion):
    candidates = {
        key: value
        for key, value in emotion.items()
        if key in reversed_azureSpeech_emotion_map
    }
    name, strength = dominant_emotion(candidates or {"neutral": 1.0})
    style = reversed_azureSpeech_emotion_map.get(name, "Default")
    degree = clamp(strength, 0.2, 0.7)
    return {"style": style, "degree": round(degree, 3)}


def gesture_metadata(emotion):
    candidates = {
        key: value
        for key, value in emotion.items()
        if key in reversed_gesture_emotion_map
    }
    name, strength = dominant_emotion(candidates or {"neutral": 1.0})
    style = reversed_gesture_emotion_map.get(name, "neutral")
    return {"style": style, "dominant_emotion": name, "strength": round(strength, 3)}


def load_script(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError("script root must be a JSON object")
    return data


def iter_dialogue(script_data, script_id):
    script = script_data[script_id]
    dialogue = script.get("dialogue", [])
    for index, turn in enumerate(dialogue, start=1):
        if not isinstance(turn, dict) or len(turn) != 1:
            raise ValueError(f"dialogue item {index} must contain exactly one speaker")
        speaker, text = next(iter(turn.items()))
        yield index, speaker, text


def output_stem(script_id, index, speaker):
    return f"{script_id}_{index:03d}_{speaker}"


def build_doctor_item(
    script_id,
    index,
    total_items,
    text,
    condition,
    audio_dir,
    gesture_dir,
    emotion_classifier=None,
):
    flags = CONDITIONS[condition]
    speaker = "doctor"
    stem = output_stem(script_id, index, speaker)
    base_emotion, emotion_source = semantic_emotion_for_turn(
        emotion_classifier,
        speaker,
        text,
        index,
        total_items,
    )
    semantic_emotion = condition_semantic_emotion(condition, base_emotion)

    item = {
        "script_id": script_id,
        "index": index,
        "speaker": speaker,
        "text": text,
        "semantic_emotion": semantic_emotion,
        "emotion_source": emotion_source,
        "condition": {
            "name": condition,
            "doctor_face_enabled": flags["face"],
            "doctor_gesture_enabled": flags["gesture"],
        },
    }

    wav_path = audio_dir / f"{stem}.wav"
    bvh_path = gesture_dir / f"{stem}.bvh"
    face_emotion = semantic_emotion.copy()
    gesture_emotion = gesture_emotion_from_semantic(base_emotion)

    item.update(
        {
            "role_in_experiment": "avatar_doctor",
            "audio": str(wav_path).replace("\\", "/"),
            "bvh": str(bvh_path).replace("\\", "/"),
            "azure": azure_metadata(semantic_emotion),
            "face_control": {
                "enabled": flags["face"],
                "ace_override_strength": 0.85 if flags["face"] else 0.0,
                "emotion": face_emotion,
            },
            "gesture_control": {
                "enabled": flags["gesture"],
                "emotion": gesture_emotion,
                **gesture_metadata(gesture_emotion),
            },
            "generation_status": {
                "audio": "exists" if wav_path.exists() else "pending",
                "bvh": "exists" if bvh_path.exists() else ("pending" if flags["gesture"] else "not_required"),
            },
        }
    )

    return item


def build_patient_turn(script_id, index, text):
    return {
        "script_id": script_id,
        "index": index,
        "speaker": "patient",
        "role_in_experiment": "user_patient",
        "input": "local_microphone",
        "expected_user_text": text,
        "metadata": None,
        "audio": None,
        "bvh": None,
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def cleanup_patient_metadata(condition_metadata_dir):
    if not condition_metadata_dir.exists():
        return []
    removed = []
    for path in condition_metadata_dir.glob("*_patient.json"):
        path.unlink()
        removed.append(path)
    return removed


def generate_metadata(args):
    script_data = load_script(args.script)
    script_ids = [args.script_id] if args.script_id else list(script_data.keys())
    selected_conditions = list(CONDITIONS.keys()) if args.condition == "all" else [args.condition]
    emotion_classifier = build_emotion_classifier(args)

    written_manifests = []
    for script_id in script_ids:
        dialogue = list(iter_dialogue(script_data, script_id))
        total_items = len(dialogue)

        for condition in selected_conditions:
            doctor_items = []
            turns = []
            condition_metadata_dir = args.metadata_dir / condition
            removed_patient_files = cleanup_patient_metadata(condition_metadata_dir)

            for index, speaker, text in dialogue:
                if speaker == "doctor":
                    item = build_doctor_item(
                        script_id=script_id,
                        index=index,
                        total_items=total_items,
                        text=text,
                        condition=condition,
                        audio_dir=args.audio_dir,
                        gesture_dir=args.gesture_dir,
                        emotion_classifier=emotion_classifier,
                    )
                    stem = output_stem(script_id, index, speaker)
                    item_path = condition_metadata_dir / f"{stem}.json"
                    write_json(item_path, item)
                    item_path_text = str(item_path).replace("\\", "/")
                    doctor_items.append(item_path_text)
                    turns.append(
                        {
                            "index": index,
                            "speaker": speaker,
                            "role_in_experiment": "avatar_doctor",
                            "metadata": item_path_text,
                            "audio": item["audio"],
                            "bvh": item["bvh"],
                        }
                    )
                elif speaker == "patient":
                    turns.append(build_patient_turn(script_id, index, text))
                else:
                    raise ValueError(f"unsupported speaker: {speaker}")

            manifest = {
                "script_id": script_id,
                "condition": condition,
                "background": script_data.get(script_id, {}).get("background"),
                "patient_role": "user",
                "doctor_role": "avatar",
                "emotion_source": "emotion_classifier" if emotion_classifier else "heuristic",
                "doctor_items": doctor_items,
                "patient_metadata": "omitted_user_microphone_turns",
                "removed_patient_metadata_count": len(removed_patient_files),
                "turns": turns,
            }
            if manifest["background"] is None:
                del manifest["background"]
            manifest_path = condition_metadata_dir / f"{script_id}_manifest.json"
            write_json(manifest_path, manifest)
            written_manifests.append(manifest_path)

    return written_manifests


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare medical script metadata for UE ACE experiments.")
    parser.add_argument("--script", type=Path, default=Path("prompt/medical/script.json"))
    parser.add_argument("--script-id", default=None)
    parser.add_argument("--condition", choices=[*CONDITIONS.keys(), "all"], default="all")
    parser.add_argument("--audio-dir", type=Path, default=Path("audio/medical"))
    parser.add_argument("--gesture-dir", type=Path, default=Path("gesture/medical"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata/medical"))
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Use emotion_classifier.py to infer semantic emotions from each dialogue line.",
    )
    parser.add_argument("--classifier-lang", default="en-US", choices=["en-US", "zh-CN"])
    return parser.parse_args()


if __name__ == "__main__":
    manifests = generate_metadata(parse_args())
    print("Wrote manifests:")
    for manifest in manifests:
        print(f"  {manifest}")
