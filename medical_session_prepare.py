import argparse
import json
import re
import time
import wave
from datetime import datetime
from pathlib import Path

from azure_speech import SpeechController
from script_medical_play import (
    CONDITIONS,
    build_patient_turn,
    build_doctor_item,
    build_emotion_classifier,
    iter_dialogue,
    load_script,
    write_json,
)


TTS_SEGMENT_GAP_SEC = 0.01
DEFAULT_TTS_MAX_CHARS = 900
FALLBACK_TTS_MAX_CHARS = 180
TTS_RETRY_COUNT = 3
TTS_RETRY_BACKOFF_SEC = 1.5
NEUTRAL_TTS_EMOTION = {
    "neutral": 1.0,
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


def split_tts_segments(text, max_chars=DEFAULT_TTS_MAX_CHARS):
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return [text]

    raw_segments = re.split(r"(?<=[.!?])\s+", text.strip())
    segments = []
    for raw_segment in raw_segments:
        if not raw_segment:
            continue
        if len(raw_segment) <= max_chars:
            segments.append(raw_segment)
            continue

        chunk = ""
        for part in re.split(r"(?<=,)\s+", raw_segment):
            if not chunk:
                chunk = part
            elif len(chunk) + 1 + len(part) <= max_chars:
                chunk = f"{chunk} {part}"
            else:
                segments.append(chunk)
                chunk = part
        if chunk:
            segments.append(chunk)
    return segments or [text]


def concatenate_wavs(source_paths, output_path, silence_sec=TTS_SEGMENT_GAP_SEC):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params = None
    frames = []

    for source_path in source_paths:
        with wave.open(str(source_path), "rb") as source:
            source_params = source.getparams()
            if params is None:
                params = source_params
            elif source_params[:3] != params[:3]:
                raise ValueError(f"cannot concatenate wav with different params: {source_path}")

            frames.append(source.readframes(source.getnframes()))
            if source_path != source_paths[-1] and silence_sec > 0:
                silence_frames = int(source.getframerate() * silence_sec)
                frames.append(b"\x00" * silence_frames * source.getnchannels() * source.getsampwidth())

    if params is None:
        raise ValueError("no wav segments to concatenate")

    with wave.open(str(output_path), "wb") as output:
        output.setnchannels(params.nchannels)
        output.setsampwidth(params.sampwidth)
        output.setframerate(params.framerate)
        output.setcomptype(params.comptype, params.compname)
        output.writeframes(b"".join(frames))


def play_wav(path):
    import winsound

    winsound.PlaySound(str(path), winsound.SND_FILENAME)


def synthesize_with_retry(speech_controller, text, output_path, retries=TTS_RETRY_COUNT):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            speech_controller.synthesis(
                NEUTRAL_TTS_EMOTION,
                text,
                str(output_path),
                is_streaming=False,
            )
            return
        except RuntimeError as error:
            last_error = error
            if attempt >= retries:
                break
            wait_sec = TTS_RETRY_BACKOFF_SEC * attempt
            print(f"Azure TTS failed on attempt {attempt}/{retries}; retrying in {wait_sec:.1f}s.")
            print(error)
            time.sleep(wait_sec)

    raise last_error


def synthesize_segments(speech_controller, segments, audio_path):
    temp_dir = audio_path.parent / ".segments" / audio_path.stem
    temp_dir.mkdir(parents=True, exist_ok=True)
    segment_paths = []

    try:
        for segment_index, segment in enumerate(segments, start=1):
            segment_path = temp_dir / f"{segment_index:02d}.wav"
            synthesize_with_retry(speech_controller, segment, segment_path)
            segment_paths.append(segment_path)

        concatenate_wavs(segment_paths, audio_path)
    finally:
        for segment_path in segment_paths:
            if segment_path.exists():
                segment_path.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass


def generate_doctor_audio(speech_controller, item, overwrite=False, max_chars=DEFAULT_TTS_MAX_CHARS):
    audio_path = Path(item["audio"])
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    if audio_path.exists() and not overwrite:
        return audio_path, "exists"

    segments = split_tts_segments(item["text"], max_chars=max_chars)
    if len(segments) == 1:
        try:
            synthesize_with_retry(speech_controller, segments[0], audio_path)
            return audio_path, "generated"
        except RuntimeError as error:
            fallback_segments = split_tts_segments(item["text"], max_chars=FALLBACK_TTS_MAX_CHARS)
            if len(fallback_segments) == 1:
                raise
            print("Azure TTS still failed; falling back to shorter synthesis segments.")
            print(error)
            segments = fallback_segments

    synthesize_segments(speech_controller, segments, audio_path)

    return audio_path, "generated"


def prepare_doctor_turn(
    script_id,
    index,
    total_items,
    text,
    args,
    emotion_classifier,
    speech_controller,
    synthesize_audio=True,
):
    item = build_doctor_item(
        script_id=script_id,
        index=index,
        total_items=total_items,
        text=text,
        condition=args.condition,
        audio_dir=args.audio_dir,
        gesture_dir=args.gesture_dir,
        emotion_classifier=emotion_classifier,
    )

    metadata_dir = args.metadata_dir / args.condition
    metadata_path = metadata_dir / f"{script_id}_{index:03d}_doctor.json"
    write_json(metadata_path, item)

    audio_path = Path(item["audio"])
    audio_status = item["generation_status"]["audio"]
    if synthesize_audio:
        audio_path, audio_status = generate_doctor_audio(
            speech_controller,
            item,
            overwrite=args.overwrite_audio,
            max_chars=args.tts_max_chars,
        )
        item["generation_status"]["audio"] = audio_status
        write_json(metadata_path, item)

    return item, metadata_path, audio_path, audio_status


def write_manifest(script_id, script_data, condition, metadata_dir, session_turns):
    condition_metadata_dir = metadata_dir / condition
    doctor_items = [
        turn["metadata"]
        for turn in session_turns
        if turn["speaker"] == "doctor"
    ]
    manifest_turns = []

    for turn in session_turns:
        if turn["speaker"] == "doctor":
            manifest_turns.append(
                {
                    "index": turn["index"],
                    "speaker": "doctor",
                    "role_in_experiment": "avatar_doctor",
                    "metadata": turn["metadata"],
                    "audio": turn["audio"],
                    "bvh": turn["bvh"],
                    "ue_anim": turn["ue_anim"],
                }
            )
            continue

        manifest_turns.append(
            build_patient_turn(script_id, turn["index"], turn["expected_user_text"])
        )

    manifest = {
        "script_id": script_id,
        "condition": condition,
        "background": script_data.get(script_id, {}).get("background"),
        "patient_role": "user",
        "doctor_role": "avatar",
        "emotion_source": "emotion_classifier" if any(
            turn.get("emotion_source") == "emotion_classifier"
            for turn in session_turns
            if turn["speaker"] == "doctor"
        ) else "heuristic",
        "doctor_items": doctor_items,
        "patient_metadata": "omitted_user_microphone_turns",
        "removed_patient_metadata_count": 0,
        "turns": manifest_turns,
    }
    if manifest["background"] is None:
        del manifest["background"]

    manifest_path = condition_metadata_dir / f"{script_id}_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run(args):
    script_data = load_script(args.script)
    script_id = args.script_id or next(iter(script_data.keys()))
    dialogue = list(iter_dialogue(script_data, script_id))
    total_items = len(dialogue)
    emotion_classifier = build_emotion_classifier(args)

    speech_controller = SpeechController(args.lang, role=args.voice)
    session_turns = []

    try:
        for index, speaker, text in dialogue:
            in_selected_range = args.start_index <= index and (
                args.end_index is None or index <= args.end_index
            )
            if speaker == "doctor":
                item, metadata_path, audio_path, audio_status = prepare_doctor_turn(
                    script_id,
                    index,
                    total_items,
                    text,
                    args,
                    emotion_classifier,
                    speech_controller,
                    synthesize_audio=in_selected_range,
                )
                print(f"\n[Doctor {index:03d}] {text}")
                print(f"Audio: {audio_path} ({audio_status})")

                if in_selected_range and not args.prepare_only and not args.no_playback:
                    play_wav(audio_path)

                session_turns.append(
                    {
                        "index": index,
                        "speaker": "doctor",
                        "text": text,
                        "metadata": str(metadata_path).replace("\\", "/"),
                        "audio": str(audio_path).replace("\\", "/"),
                        "bvh": item["bvh"],
                        "ue_anim": item["ue_anim"],
                        "semantic_emotion": item["semantic_emotion"],
                        "emotion_source": item["emotion_source"],
                    }
                )
                continue

            if speaker != "patient":
                raise ValueError(f"unsupported speaker: {speaker}")

            patient_turn = {
                "index": index,
                "speaker": "patient",
                "expected_user_text": text,
                "input": "local_microphone",
                "recognize_method": args.recognize_method,
                "hold_key": args.hold_key if args.recognize_method == "push-to-talk" else None,
            }

            if in_selected_range and args.show_patient_script:
                print(f"\n[Patient {index:03d}] Please read aloud:")
                print(text)

            if args.prepare_only or not in_selected_range:
                session_turns.append(patient_turn)
                continue

            if args.recognize_method == "push-to-talk":
                recognized_text, recognize_time = speech_controller.recognize_push_to_talk(args.hold_key)
            else:
                print("\n[Patient] Please answer with the local microphone.")
                recognized_text, recognize_time = speech_controller.recognize(args.recognize_method)

            patient_turn.update(
                {
                    "recognized_text": recognized_text,
                    "recognize_time_sec": round(recognize_time, 3),
                }
            )
            session_turns.append(patient_turn)

    finally:
        speech_controller.close()

    session_log = {
        "script_id": script_id,
        "condition": args.condition,
        "lang": args.lang,
        "voice": args.voice,
        "recognize_method": args.recognize_method,
        "hold_key": args.hold_key if args.recognize_method == "push-to-talk" else None,
        "mode": "prepare_only" if args.prepare_only else "local_microphone_loop",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "turns": session_turns,
    }
    args.session_log.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.session_log, session_log)
    manifest_path = write_manifest(
        script_id,
        script_data,
        args.condition,
        args.metadata_dir,
        session_turns,
    )
    print(f"Manifest: {manifest_path}")
    print(f"\nSession log: {args.session_log}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate doctor audio and run a minimal doctor-avatar / patient-mic loop."
    )
    parser.add_argument("--script", type=Path, default=Path("prompt/medical/script.json"))
    parser.add_argument("--script-id", default=None)
    parser.add_argument("--condition", choices=CONDITIONS.keys(), default="baseline")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="Start synthesizing/playing from this dialogue turn index, e.g. 5 for script2_005_doctor.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Stop synthesizing/playing after this dialogue turn index.",
    )
    parser.add_argument("--audio-dir", type=Path, default=Path("audio/medical"))
    parser.add_argument("--gesture-dir", type=Path, default=Path("gesture/medical"))
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata/medical"))
    parser.add_argument(
        "--session-log",
        type=Path,
        default=Path("metadata/medical/session_logs/latest.json"),
    )
    parser.add_argument("--lang", default="en-US")
    parser.add_argument("--voice", default=None)
    parser.add_argument(
        "--recognize-method",
        choices=["push-to-talk", "once", "continuous"],
        default="push-to-talk",
    )
    parser.add_argument("--hold-key", default="space")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument(
        "--tts-max-chars",
        type=int,
        default=DEFAULT_TTS_MAX_CHARS,
        help="Synthesize each doctor turn as one wav unless it exceeds this length. Use 0 to never split.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-playback", action="store_true")
    parser.set_defaults(show_patient_script=True)
    parser.add_argument("--show-patient-script", dest="show_patient_script", action="store_true")
    parser.add_argument("--hide-patient-script", dest="show_patient_script", action="store_false")
    parser.add_argument(
        "--use-classifier",
        action="store_true",
        help="Use emotion_classifier.py to infer doctor semantic emotions.",
    )
    parser.add_argument("--classifier-lang", default="en-US", choices=["en-US", "zh-CN"])
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
