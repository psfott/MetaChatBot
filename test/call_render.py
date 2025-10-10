import subprocess

if __name__ == '__main__':
    cmd = [r"D:\Blender\blender.exe", "-b", r"F:\Github Repo\blendshapes-visualization\resources\blendshapes.blend",
           "-P", "./render.py", "--",
           r"C:\Users\pitbull\OneDrive\Python Programs\MetaChatBot\test\blendshape\bs"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line.strip())
