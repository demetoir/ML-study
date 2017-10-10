"""
image renaming(and saving new folder)
ordered images -> generate tile image(input: column size)
"""

import os
from PIL import Image
from os.path import join

logo_img_root = '/home/ml-lab/Downloads/GAN'


def rename_file(old_path):

    rename_dir = join(logo_img_root, '5klogos(rename)')
    if not os.path.exists(rename_dir):
        os.mkdir(rename_dir)
        print("Make dir 'new(rename)'")
    elif os.listdir(rename_dir):
        print("Already exist image file in 'rename' folder")
        return rename_dir   # file exist

    for i, img_name in enumerate(os.listdir(old_path), 1):
        img_path = rename_dir + '/' + str(i) + '.png'
        os.rename(old_path + '/' + img_name, img_path)
        print(img_name, img_path)
        i += 1

    return rename_dir


def gen_image_tile(image_list, size_column=20):
    image = Image.open(image_list[0])
    x = image.size[0]
    y = image.size[0]
    length = len(image_list)

    tile_img = Image.new("RGBA", (int(size_column * x), int((((length - 1) / size_column) + 1) * y)))

    print("x,y:", x, y, int(size_column), int(((length - 1) // size_column) + 1), (
        int(size_column * x), int((((length - 1) // size_column) + 1) * y)))

    for i, image_path in enumerate(image_list, 0):
        image = Image.open(image_path)
        box = (0, 0, x, y)
        cutting = image.crop(box)
        print("Process:", image_path, (i // size_column), (i % size_column), box,
              (x * (i // size_column), y * (i % size_column)))

        tile_img.paste(cutting, (int(x * (i % size_column)), int(y * (i // size_column))))
        i = i + 1

    tile_img.save(join(logo_img_root, 'tile_logo_%dcolumn.png' % size_column), "PNG")


if __name__ == '__main__':
    path = join(logo_img_root, '5klogos')

    new_dir = rename_file(path)

    image_list = [join(new_dir, image_name) for image_name in os.listdir(new_dir)]

    # column_size = 40
    gen_image_tile(sorted(image_list)) #, column_size)
    pass
