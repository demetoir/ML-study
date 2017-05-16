import os
path = os.path
import subprocess

DEFAULT_LOGDIR = ""

# tensorboard --logdir=C:\Users\yujun\Documents\ML-image-classify\tensorboard_log
dir_log = os.path.join("C:\\Users\\yujun\\Documents\\ML-image-classify", "tensorboard_log")
cmd = "tensorboard --logdir=" + dir_log

if __name__ == '__main__':
    print("tensorboard start")
    p_tensorboard = subprocess.Popen(cmd)
    print("http://127.0.0.1:6006/")

    while True:
        print("exit y/n?")
        if input() == 'y':
            break

    p_tensorboard.kill()
    print("tensorboard killed")
    pass
