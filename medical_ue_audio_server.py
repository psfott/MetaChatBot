import argparse
import json
import time
import wave
from pathlib import Path

from async_server import Server


ACE_EMOTION_OVERRIDE_KEYS = {
    "amazement": "amazement",
    "anger": "anger",
    "cheekiness": "cheekiness",
    "disgust": "disgust",
    "fear": "fear",
    "grief": "grief",
    "joy": "joy",
    "out_of_breath": "out_of_breath",
    "pain": "pain",
    "sadness": "sadness",
}


ACE_EMOTION_OVERRIDE_DEFAULTS = {
    "amazement": 0.0,
    "anger": 0.0,
    "cheekiness": 0.0,
    "disgust": 0.0,
    "fear": 0.0,
    "grief": 0.0,
    "joy": 0.0,
    "out_of_breath": 0.0,
    "pain": 0.0,
    "sadness": 0.0,
}


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def wav_duration(path):
    with wave.open(str(path), "rb") as wav_file:
        return wav_file.getnframes() / wav_file.getframerate()


def dominant_emotion(semantic_emotion, include_neutral=True):
    candidates = semantic_emotion.items()
    if not include_neutral:
        candidates = [(key, value) for key, value in candidates if key != "neutral"]
        if not candidates:
            return "neutral", 0.0
    return max(candidates, key=lambda item: item[1])


def build_expression(expression_mode, semantic_emotion, max_strength):
    if expression_mode == "neutral":
        return {
            "mode": "neutral",
            "dominant_emotion": "neutral",
            "strength": 0.0,
        }

    emotion, strength = dominant_emotion(semantic_emotion, include_neutral=False)
    scaled_strength = min(max(strength, 0.0), max_strength)

    return {
        "mode": "emotional",
        "dominant_emotion": emotion,
        "strength": round(scaled_strength, 3),
    }


def clamp01(value):
    return min(max(float(value), 0.0), 1.0)


def build_ace_emotion_parameters(
    expression_mode,
    semantic_emotion,
    max_expression_strength,
    overall_emotion_strength,
    emotion_override_strength,
    detected_emotion_contrast,
    max_detected_emotions,
    detected_emotion_smoothing,
):
    if expression_mode == "neutral":
        return {
            "overall_emotion_strength": 0.0,
            "detected_emotion_contrast": detected_emotion_contrast,
            "max_detected_emotions": max_detected_emotions,
            "detected_emotion_smoothing": detected_emotion_smoothing,
            "enable_emotion_override": False,
            "emotion_override_strength": 0.0,
            "emotion_overrides": ACE_EMOTION_OVERRIDE_DEFAULTS,
        }

    emotion_overrides = ACE_EMOTION_OVERRIDE_DEFAULTS.copy()
    for source_name, target_name in ACE_EMOTION_OVERRIDE_KEYS.items():
        source_value = semantic_emotion.get(source_name, 0.0)
        emotion_overrides[target_name] = round(
            clamp01(source_value) * max_expression_strength,
            3,
        )

    has_override = any(value > 0.0 for value in emotion_overrides.values())
    return {
        "overall_emotion_strength": round(clamp01(overall_emotion_strength), 3),
        "detected_emotion_contrast": round(float(detected_emotion_contrast), 3),
        "max_detected_emotions": int(max_detected_emotions),
        "detected_emotion_smoothing": round(clamp01(detected_emotion_smoothing), 3),
        "enable_emotion_override": has_override,
        "emotion_override_strength": round(clamp01(emotion_override_strength), 3)
        if has_override
        else 0.0,
        "emotion_overrides": emotion_overrides,
    }


def audio_message(item, metadata_path, args):
    audio_path = Path(item["audio"]).resolve()
    semantic_emotion = item.get("semantic_emotion", {"neutral": 1.0})
    duration = wav_duration(audio_path)
    expression = build_expression(
        args.expression_mode,
        semantic_emotion,
        args.max_expression_strength,
    )
    ace_emotion_parameters = build_ace_emotion_parameters(
        args.expression_mode,
        semantic_emotion,
        args.max_expression_strength,
        args.ace_overall_emotion_strength,
        args.ace_emotion_override_strength,
        args.ace_detected_emotion_contrast,
        args.ace_max_detected_emotions,
        args.ace_detected_emotion_smoothing,
    )

    return {
        "type": "doctor_audio",
        "protocol": "metachatbot.medical.v1",
        "turn_index": item["index"],
        "speaker": "doctor",
        "condition": item["condition"]["name"],
        "text": item["text"],
        "metadata_path": str(Path(metadata_path).resolve()).replace("\\", "/"),
        "audio_path": str(audio_path).replace("\\", "/"),
        "audio_uri": audio_path.as_uri(),
        "duration_sec": round(duration, 3),
        "audio_source": {
            "kind": "wav_file",
            "path": str(audio_path).replace("\\", "/"),
            "uri": audio_path.as_uri(),
            "duration_sec": round(duration, 3),
        },
        "ace": {
            "enabled": True,
            "use_audio_for_lipsync": True,
            "expression": expression,
            "emotion_parameters": ace_emotion_parameters,
        },
        "gesture": {
            "enabled": False,
        },
    }


def patient_prompt_message(turn, condition_name):
    return {
        "type": "patient_prompt",
        "protocol": "metachatbot.medical.v1",
        "turn_index": turn["index"],
        "speaker": "patient",
        "condition": condition_name,
        "text": turn["expected_user_text"],
        "expected_user_text": turn["expected_user_text"],
        "interaction": {
            "display_text": True,
            "push_to_talk": True,
            "hold_key": "T",
            "done_signal": f"Patient released:{turn['index']}",
        },
    }


def wait_for_client(server, timeout_sec):
    start = time.time()
    while not server.clients:
        if timeout_sec is not None and time.time() - start > timeout_sec:
            raise TimeoutError("UE TCP client did not connect before timeout.")
        time.sleep(0.1)


def send_json(server, payload):
    message = json.dumps(payload, ensure_ascii=False) + "\n"
    server.send_data(message)
    print(f"Sent to UE: {message.strip()}")


def wait_for_done_or_duration(
    server,
    turn_index,
    duration_sec,
    wait_ack,
    duration_padding_sec,
    first_turn_padding_sec,
):
    fallback_sec = duration_sec + duration_padding_sec
    if turn_index == 1:
        fallback_sec += first_turn_padding_sec

    if not wait_ack:
        print(f"Waiting {fallback_sec:.2f}s before next turn.")
        time.sleep(fallback_sec)
        return

    print(f"Waiting for UE ack: Cur index:{turn_index}")
    start = time.time()
    while server.done_index != turn_index:
        if time.time() - start > fallback_sec + 10.0:
            print("UE ack timeout; continuing by duration fallback.")
            return
        time.sleep(0.05)


def wait_for_patient_done(server, turn_index, timeout_sec):
    server.done_index = None
    server.patient_released_index = None
    print(f"Waiting for patient turn {turn_index}; release T in UE to continue.")
    start = time.time()
    while server.patient_released_index != turn_index:
        if timeout_sec is not None and time.time() - start > timeout_sec:
            print("Patient turn timeout; continuing.")
            return
        time.sleep(0.05)
    print(f"Patient release ack received: Patient released:{turn_index}")


def run(args):
    manifest = read_json(args.manifest)
    server = None
    condition_name = Path(args.manifest).parent.name
    print(
        "Medical UE audio server config: "
        f"manifest={args.manifest}, "
        f"expression_mode={args.expression_mode}, "
        f"wait_ack={args.wait_ack}, "
        f"pause_on_patient={args.pause_on_patient}, "
        f"wait_patient_from_ue={args.wait_patient_from_ue}, "
        f"patient_prompt_delay={args.patient_prompt_delay}, "
        f"doctor_only={args.doctor_only}"
    )

    if not args.dry_run:
        server = Server(args.host, args.port)
        server.daemon = True
        server.start()
        print(f"Waiting for UE TCP client on {args.host}:{args.port} ...")
        wait_for_client(server, args.connect_timeout)

    try:
        for turn in manifest["turns"]:
            if turn["speaker"] == "patient":
                if args.doctor_only:
                    continue

                print(f"\n[Patient {turn['index']:03d}] Please read aloud:")
                print(turn["expected_user_text"])
                prompt_payload = patient_prompt_message(turn, condition_name)
                if args.dry_run:
                    print(json.dumps(prompt_payload, indent=2, ensure_ascii=False))
                    continue

                if server is not None and args.send_patient_prompt:
                    if args.patient_prompt_delay > 0.0:
                        print(f"Waiting {args.patient_prompt_delay:.2f}s before sending patient prompt.")
                        time.sleep(args.patient_prompt_delay)
                    send_json(server, prompt_payload)
                if args.pause_on_patient:
                    if server is not None and args.wait_patient_from_ue:
                        wait_for_patient_done(server, turn["index"], args.patient_timeout)
                    else:
                        input("Press Enter after the patient turn...")
                continue

            item = read_json(turn["metadata"])
            payload = audio_message(item, turn["metadata"], args)

            print(f"\n[Doctor {item['index']:03d}] {item['text']}")
            if args.dry_run:
                print(json.dumps(payload, indent=2, ensure_ascii=False))
                continue

            send_json(server, payload)
            wait_for_done_or_duration(
                server,
                item["index"],
                payload["duration_sec"],
                args.wait_ack,
                args.duration_padding,
                args.first_turn_padding,
            )

    finally:
        if server is not None:
            server.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Send prepared medical doctor audio to UE/ACE over TCP.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )
    parser.add_argument("--emotion", action="store_true", help="Use the face condition with emotional ACE expressions.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--connect-timeout", type=float, default=None)
    parser.add_argument("--expression-mode", choices=["neutral", "emotional"], default="neutral")
    parser.add_argument("--max-expression-strength", type=float, default=1.0)
    parser.add_argument("--ace-overall-emotion-strength", type=float, default=0.95)
    parser.add_argument("--ace-emotion-override-strength", type=float, default=1.0)
    parser.add_argument("--ace-detected-emotion-contrast", type=float, default=1.4)
    parser.add_argument("--ace-max-detected-emotions", type=int, default=3)
    parser.add_argument("--ace-detected-emotion-smoothing", type=float, default=0.7)
    parser.add_argument("--wait-ack", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration-padding", type=float, default=2.0)
    parser.add_argument("--first-turn-padding", type=float, default=5.0)
    parser.add_argument("--send-patient-prompt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patient-prompt-delay", type=float, default=1.2)
    parser.add_argument("--pause-on-patient", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wait-patient-from-ue", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patient-timeout", type=float, default=None)
    parser.add_argument("--doctor-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.manifest is None:
        if args.emotion:
            args.manifest = Path("metadata/medical/face/script2_manifest.json")
            args.expression_mode = "emotional"
        else:
            args.manifest = Path("metadata/medical/baseline/script2_manifest.json")
            args.expression_mode = "neutral"

    return args


if __name__ == "__main__":
    run(parse_args())
