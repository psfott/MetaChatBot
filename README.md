# MetaChatBot Medical Experiment

This repository contains the Python-side pipeline for the medical dialogue experiment and the helper scripts used to prepare body gesture animations for Unreal Engine.

The current experiment loop is:

```text
medical script metadata
-> doctor wav audio
-> optional speech2gesture BVH
-> UE BVH import to LaForge skeleton
-> IK Retarget to NVB_female MetaHuman
-> Python TCP sends doctor audio + gesture AnimSequence path
-> UE plays ACE lipsync/audio and body gesture together
```

## Main Runtime Scripts

### `medical_ue_audio_server.py`

Main script used during the UE demo / experiment.

It reads a medical manifest and sends turn-by-turn TCP messages to Unreal Engine:

- doctor audio turns
- patient prompt turns
- ACE expression settings
- optional gesture animation paths

Gesture-enabled run:

```powershell
python medical_ue_audio_server.py --manifest metadata/medical/gesture/script2_manifest.json --expression-mode neutral --gesture-mode auto
```

Baseline run without gesture:

```powershell
python medical_ue_audio_server.py --manifest metadata/medical/baseline/script2_manifest.json --expression-mode neutral --gesture-mode off
```

Dry-run to inspect exactly what JSON will be sent:

```powershell
python medical_ue_audio_server.py --manifest metadata/medical/gesture/script2_manifest.json --expression-mode neutral --gesture-mode auto --dry-run
```

Important payload field for UE gesture playback:

```json
"gesture": {
  "enabled": true,
  "source": "ue_anim",
  "ue_anim": "/Game/MetaHumans/NVB_female/Animations/Retargeted/RTG_script2_001_doctor_NVB.RTG_script2_001_doctor_NVB"
}
```

### `medical_session_prepare.py`

Prepares a medical script condition.

Typical responsibilities:

- builds or refreshes medical metadata
- ensures doctor audio paths and durations are recorded
- prepares condition-specific manifest files

Example:

```powershell
python medical_session_prepare.py --script-id script2 --condition gesture --prepare-only
```

Regenerate/overwrite audio if needed:

```powershell
python medical_session_prepare.py --script-id script2 --condition gesture --prepare-only --overwrite-audio
```

### `medical_gesture_prepare.py`

Generates BVH body gesture files for doctor turns using the local `speech2gesture` pipeline.

Input:

```text
metadata/medical/gesture/script2_manifest.json
```

Output:

```text
gesture/medical/script2_001_doctor.bvh
gesture/medical/script2_003_doctor.bvh
...
metadata/medical/gesture/script2_001_doctor.json
...
```

Run:

```powershell
python medical_gesture_prepare.py --manifest metadata/medical/gesture/script2_manifest.json --overwrite
```

Resume from a later doctor turn:

```powershell
python medical_gesture_prepare.py --manifest metadata/medical/gesture/script2_manifest.json --start-index 5 --overwrite
```

## Supporting Python Modules

### `azure_speech.py`

Handles Azure speech synthesis for doctor audio generation.

Used indirectly by preparation scripts when audio needs to be created or regenerated.

### `emotion_classifier.py`

Classifies or maps text emotion information used by the experiment conditions.

Its output can affect:

- ACE facial expression parameters
- gesture style metadata
- `dominant_emotion`
- gesture strength fields

### `speech2gesture/`

Local gesture generation code.

Important files:

```text
speech2gesture/gesture_generator.py
speech2gesture/data_pipeline.py
speech2gesture/bvh_loader.py
speech2gesture/utils.py
```

This part is responsible for turning audio/text/style inputs into BVH motion.

## Medical Data Layout

### `metadata/medical/`

Condition-specific dialogue metadata and manifests.

Typical structure:

```text
metadata/medical/baseline/
metadata/medical/gesture/
metadata/medical/face/
metadata/medical/full/
```

Important gesture files:

```text
metadata/medical/gesture/script2_manifest.json
metadata/medical/gesture/script2_001_doctor.json
metadata/medical/gesture/script2_003_doctor.json
metadata/medical/gesture/script2_005_doctor.json
metadata/medical/gesture/script2_007_doctor.json
metadata/medical/gesture/script2_009_doctor.json
metadata/medical/gesture/script2_011_doctor.json
```

To change only one doctor turn's UE animation, edit that turn's JSON:

```json
"ue_anim": "/Game/MetaHumans/NVB_female/Animations/UE5Actions/NVB_NM0001.NVB_NM0001",
"ue_asset_path": "/Game/MetaHumans/NVB_female/Animations/UE5Actions/NVB_NM0001.NVB_NM0001"
```

Use full UE object paths:

```text
/Game/Folder/AssetName.AssetName
```

### `audio/medical/`

Doctor audio files used by ACE / Audio2Face.

Example:

```text
audio/medical/script2_001_doctor.wav
audio/medical/script2_003_doctor.wav
```

### `gesture/medical/`

Generated BVH files from `medical_gesture_prepare.py`.

Example:

```text
gesture/medical/script2_001_doctor.bvh
gesture/medical/script2_003_doctor.bvh
```

These BVH files are not played directly in the current runtime. They are imported into UE and retargeted into MetaHuman-compatible `AnimSequence` assets first.

## UE Helper Scripts

UE helper scripts live in `tools/`. Most `ue_*.py` scripts are intended to run inside Unreal Editor Python, not normal system Python.

### `tools/ue_import_medical_bvh.py`

Imports generated BVH files into the UE project with the BVH plugin.

Pipeline stage:

```text
gesture/medical/script2_*_doctor.bvh
-> BVHPlugin
-> /Game/BVH/MedicalGeneratedV2/script2_*_doctor
```

Target skeleton:

```text
/Game/BVH/LaForgeMale_Skeleton
```

### `tools/ue_retarget_laforge_to_nvb.py`

Retargets imported LaForge animations to the `NVB_female` MetaHuman body.

Pipeline stage:

```text
/Game/BVH/MedicalGeneratedV2/script2_*_doctor
-> /Game/MetaHumans/Retargeter/RTG_LaForge_to_NVB_female
-> /Game/MetaHumans/NVB_female/Animations/Retargeted/RTG_script2_*_doctor_NVB
```

Key UE assets:

```text
Source IK Rig: /Game/BVH/LeForgeIK
Source Mesh:   /Game/BVH/LaForgeMale
Target IK Rig: /Game/MetaHumans/Common/Common/IK_metahuman
Target Mesh:   /Game/MetaHumans/NVB_female/Body/SKM_NVB_female_BodyMesh
```

### `tools/ue_validate_retargeted_anim.py`

Checks whether expected retargeted animation assets exist and can be loaded.

Useful after batch retargeting.

### `tools/ue_list_mixamo_assets.py`

Lists Mixamo-related UE assets.

Used while preparing sitting / idle animations.

### `tools/ue_retarget_mixamo_sitting_to_nvb.py`

Creates or uses a Mixamo-to-NVB retarget pipeline for sitting base animation.

Important output:

```text
/Game/MetaHumans/NVB_female/Animations/POSE_NVB_female_SittingBase
```

### `tools/ue_retarget_ue5_actions_to_nvb.py`

Prepares retargeting from UE5 Manny/Quinn action animations to `NVB_female`.

Current UE5Actions examples:

```text
/Game/MetaHumans/NVB_female/Animations/UE5Actions/NVB_NM0001.NVB_NM0001
/Game/MetaHumans/NVB_female/Animations/UE5Actions/NVB_AS_FN_NM0001.NVB_AS_FN_NM0001
```

## Unreal Engine Runtime Setup

The UE project is:

```text
G:/UE_Project/medical_chatbot
```

Important C++ files in the UE project:

```text
Source/medical_chatbot/MedicalTcpClientComponent.h
Source/medical_chatbot/MedicalTcpClientComponent.cpp
Source/medical_chatbot/MedicalAceEmotionLibrary.h
Source/medical_chatbot/MedicalAceEmotionLibrary.cpp
Source/medical_chatbot/MedicalGestureLibrary.h
Source/medical_chatbot/MedicalGestureLibrary.cpp
```

Responsibilities:

- `MedicalTcpClientComponent`: TCP messages between UE and Python.
- `MedicalAceEmotionLibrary`: parses ACE expression parameters from medical JSON.
- `MedicalGestureLibrary`: parses `gesture.ue_anim` and loads the UE `AnimSequence`.

Blueprint runtime flow:

```text
BP_MedicalTcpBridge
-> On Doctor Audio Received
-> Make Ace Emotion from Medical Json
-> send wav to ACE / Audio2Face
-> Try Load Medical Gesture Anim
-> set ABP_NVB_female_DoctorBody.GestureAnim
-> set ABP_NVB_female_DoctorBody.PlayGesture = true
```

At doctor audio completion:

```text
Reset Doctor Gesture
-> PlayGesture = false
-> Send Cur Index
-> Python sends next patient prompt
```

The current body animation blueprint is:

```text
ABP_NVB_female_DoctorBody
```

It uses:

```text
GestureAnim
-> Sequence Player
-> Blend Poses by Bool
-> Output Pose
```

`PlayGesture` controls whether the gesture animation is active.

## Full Gesture Preparation Pipeline

Run this when gesture BVH and UE retargeted animations need to be regenerated.

1. Prepare metadata and audio:

```powershell
python medical_session_prepare.py --script-id script2 --condition gesture --prepare-only
```

2. Generate BVH:

```powershell
python medical_gesture_prepare.py --manifest metadata/medical/gesture/script2_manifest.json --overwrite
```

3. Import BVH into UE:

```text
Run tools/ue_import_medical_bvh.py inside Unreal Editor Python.
```

Expected UE assets:

```text
/Game/BVH/MedicalGeneratedV2/script2_001_doctor
/Game/BVH/MedicalGeneratedV2/script2_003_doctor
...
```

4. Retarget LaForge to NVB_female:

```text
Run tools/ue_retarget_laforge_to_nvb.py inside Unreal Editor Python.
```

Expected UE assets:

```text
/Game/MetaHumans/NVB_female/Animations/Retargeted/RTG_script2_001_doctor_NVB
/Game/MetaHumans/NVB_female/Animations/Retargeted/RTG_script2_003_doctor_NVB
...
```

5. Confirm metadata points to the correct UE animation paths:

```powershell
python medical_ue_audio_server.py --manifest metadata/medical/gesture/script2_manifest.json --expression-mode neutral --gesture-mode auto --dry-run
```

6. Run the UE experiment:

```powershell
python medical_ue_audio_server.py --manifest metadata/medical/gesture/script2_manifest.json --expression-mode neutral --gesture-mode auto
```

## Minimal Python Dependencies

For the medical experiment subset, install:

```powershell
pip install -r mini_requirements.txt
```

UE Python scripts run in Unreal Editor Python and do not need an `unreal` pip package.

## Notes

- The current runtime uses offline retargeted `AnimSequence` assets, not runtime BVH import.
- If an animation does not change in UE, first check the Python `--dry-run` output and confirm `gesture.ue_anim` is the path you expect.
- If UE fails to load a gesture, check the UE object path format:

```text
/Game/Folder/AssetName.AssetName
```

- If patient speech starts while the doctor is still holding the previous gesture, make sure `Reset Doctor Gesture` runs before `Send Cur Index`.
