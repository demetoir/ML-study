import os
import sys

path = os.path
import subprocess
import webbrowser

DEFAULT_LOGDIR = ""

# tensorboard --logdir=C:\Users\yujun\Documents\ML-image-classify\tensorboard_log
dir_log = os.path.join("C:\\Users\\yujun\\Documents\\ML-image-classify", "tensorboard_log")
cmd = "tensorboard --logdir=" + dir_log
url = "http://127.0.0.1:6006/"

if __name__ == '__main__':
    print("tensorboard start")
    p_tensorboard = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    webbrowser.open(url)

    while True:
        print("exit y/n?")
        if input() == 'y':
            break

    p_tensorboard.kill()
    print("tensorboard killed")
    pass
