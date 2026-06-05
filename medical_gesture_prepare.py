import argparse
import json
import random
import time
from pathlib import Path

from speech2gesture.gesture_generator import FILE, GestureGenerator


DEFAULT_OPTION_FILE = Path("model/options.json")
DEFAULT_MANIFEST = Path("metadata/medical/gesture/script2_manifest.json")


def read_json(path):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def should_include_turn(index, start_index, end_index):
    if start_index is not None and index < start_index:
        return False
    if end_index is not None and index > end_index:
        return False
    return True


def doctor_turns(manifest, start_index=None, end_index=None):
    for turn in manifest["turns"]:
        if turn.get("speaker") != "doctor":
            continue
        if should_include_turn(turn["index"], start_index, end_index):
            yield turn


def resolve_repo_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def build_generator(args):
    return GestureGenerator(
        str(args.option_file),
        temperature=args.temperature,
        seed=args.seed,
        use_gpu=args.use_gpu,
        use_preload=True,
        use_thread=False,
        is_blocking=True,
        load_size=args.load_size,
        generating_mode=FILE,
    )


def optional_repo_path(path_value):
    if path_value is None:
        return None
    return resolve_repo_path(path_value)


def generate_turn(generator, turn, args, first_pose=None):
    metadata_path = resolve_repo_path(turn["metadata"])
    item = read_json(metadata_path)
    gesture_control = item.get("gesture_control", {})
    audio_path = resolve_repo_path(item["audio"])
    bvh_path = resolve_repo_path(item["bvh"])

    # Gesture pipeline, generation stage:
    # 1. Use the already synthesized doctor wav as the speech driver.
    # 2. Use gesture_control.emotion/style from metadata to choose the ZeroEGGS style example.
    # 3. Write the generated BVH to item["bvh"], which UE later receives over TCP.
    # 4. medical_ue_audio_server.py will expose this file as gesture.bvh_path for BVHPlugin.
    if not gesture_control.get("enabled", False):
        return {
            "turn_index": item["index"],
            "status": "skipped",
            "reason": "gesture_control.disabled",
            "bvh": str(bvh_path).replace("\\", "/"),
        }, first_pose

    if not audio_path.exists():
        raise FileNotFoundError(f"Missing doctor audio for turn {item['index']}: {audio_path}")

    if bvh_path.exists() and not args.overwrite:
        return {
            "turn_index": item["index"],
            "status": "exists",
            "audio": str(audio_path).replace("\\", "/"),
            "bvh": str(bvh_path).replace("\\", "/"),
        }, bvh_path if args.chain_first_pose else first_pose

    bvh_path.parent.mkdir(parents=True, exist_ok=True)
    emotion = gesture_control.get("emotion", {"neutral": 1.0})
    print(
        f"[Gesture {item['index']:03d}] audio={audio_path} "
        f"style={gesture_control.get('style', 'neutral')} -> {bvh_path}"
    )
    audio_length, _ = generator.generate(
        str(audio_path),
        str(bvh_path),
        encoding_file=str(optional_repo_path(args.style_bvh)) if args.style_bvh else None,
        emotion_dict=emotion,
        first_pose=first_pose if args.chain_first_pose else None,
        verbose_level=args.verbose,
    )

    status = "generated" if bvh_path.exists() else "missing_after_generate"
    item.setdefault("generation_status", {})["bvh"] = "exists" if bvh_path.exists() else "failed"
    write_json(metadata_path, item)
    return {
        "turn_index": item["index"],
        "status": status,
        "audio_sec": round(audio_length, 3),
        "audio": str(audio_path).replace("\\", "/"),
        "bvh": str(bvh_path).replace("\\", "/"),
    }, bvh_path if args.chain_first_pose and bvh_path.exists() else first_pose


def run(args):
    manifest = read_json(args.manifest)
    turns = list(doctor_turns(manifest, args.start_index, args.end_index))
    if not turns:
        raise RuntimeError("No doctor turns matched the manifest/index filter.")

    print(
        f"Gesture prepare: manifest={args.manifest}, "
        f"turns={len(turns)}, option_file={args.option_file}, use_gpu={args.use_gpu}"
    )
    if args.dry_run:
        for turn in turns:
            item = read_json(resolve_repo_path(turn["metadata"]))
            print(
                f"[DRY] turn={item['index']:03d} "
                f"audio={item.get('audio')} bvh={item.get('bvh')} "
                f"gesture_enabled={item.get('gesture_control', {}).get('enabled')}"
            )
        return []

    generator = build_generator(args)
    results = []
    first_pose = resolve_repo_path(args.first_pose) if args.first_pose else None
    for turn in turns:
        result, first_pose = generate_turn(generator, turn, args, first_pose=first_pose)
        results.append(result)

    print("Gesture generation summary:")
    for result in results:
        print(f"  turn={result['turn_index']:03d} status={result['status']} bvh={result['bvh']}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Generate medical doctor BVH gestures from prepared audio.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--option-file", type=Path, default=DEFAULT_OPTION_FILE)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--style-bvh",
        type=Path,
        default=None,
        help="Use a specific BVH file as the gesture style exemplar instead of sampling by emotion.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--first-pose", type=Path, default=None)
    parser.add_argument("--chain-first-pose", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(time.time()) ^ random.randint(0, 2**16)
    return args


if __name__ == "__main__":
    run(parse_args())
