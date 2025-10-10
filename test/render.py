import subprocess
import sys

import bpy
from pathlib import Path
import os
import numpy as np
import shutil

if __name__ == "__main__":
    dir = sys.argv[-1]
    blendshape_names = np.load("./blendshape/blendshape_names.npy")
    print(blendshape_names)

    output_dir = Path(dir).parent / "blender_renders"
    audio_dir = Path(dir).parent / "audio"
    output_dir.mkdir(exist_ok=True)

    for file in os.listdir(dir):

        img_dir = output_dir / "imgs"
        img_dir.mkdir(exist_ok=True)
        print(img_dir)

        if file.endswith(".npy"):
            coeffs = np.load(os.path.join(dir, file))
            # reference the active object
            o = bpy.context.active_object

            for k in blendshape_names[1:]:
                o.data.shape_keys.key_blocks[k].value = 0

            for i in range(coeffs.shape[0]):
                named_coeffs = dict(zip(blendshape_names[1:], coeffs[i][1:]))
                for k, v in named_coeffs.items():
                    o.data.shape_keys.key_blocks[k].value = v

                bpy.context.scene.render.filepath = str(img_dir / f"out_{i}.jpg")
                bpy.ops.render.render(write_still=True)

            based_name = file.split(".")[0]

            input_audio_name = based_name + ".wav"

            output_file_name = based_name + ".mp4"

            output_file = output_dir / output_file_name

            print(audio_dir / input_audio_name)

            cmds = ['ffmpeg', "-y",
                    '-i', f'{img_dir / "out_%d.png"}',  # 输入文件模板
                    "-i", f"{audio_dir / input_audio_name}",
                    '-start_number', '0',  # 开始编号
                    "-vcodec", "libx264",
                    "-r", "24",
                    '-c:a', 'aac',  # 音频编码器
                    f'{output_file}'  # 输出文件
                    ]

            process = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            for line in process.stdout:
                print(line)

            print(f"Video file {output_file} ready.")
            shutil.rmtree(img_dir)
