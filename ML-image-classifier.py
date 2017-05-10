import os
import numpy
from PIL import Image
import tensorflow as tf

file_name_format = "data_batch_%d"
folder_name = "cifar-10-batches-py"
batches_meta_file_name = "batches.meta"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def showimage(raw_data):
    r, g, b = raw_data[:1024], raw_data[1024:1024 * 2], raw_data[1024 * 2:1024 * 3]
    rgb_list = list(zip(r, g, b))
    for x in range(32):
        for y in range(32):
            img.putpixel((x, y), rgb_list[y * 32 + x])
    img.show()


if __name__ == '__main__':
    meta = unpickle(os.path.join(folder_name, batches_meta_file_name))
    label_names = meta[b'label_names']

    batch_number = 1
    d = unpickle(os.path.join(folder_name, file_name_format % batch_number))
    batch_label = d[b'batch_label']
    data = d[b'data']
    filenames = d[b'filenames']
    labels = d[b'labels']

    img_mode = 'RGB'
    img_size = (32, 32)
    img = Image.new(img_mode, img_size)

    for i in range(100):
        print("number = %4d, name = %s" % (i, filenames[i]))
        # showimage(data[i])
