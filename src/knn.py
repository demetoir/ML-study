import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import random
import colorsys


def get_colors(num_colors):
    """
    Function to generate a list of randomly generated colors
    The function first generates 256 different colors and then
    we randomly select the number of colors required from it
    num_colors        -> Number of colors to generate
    colors            -> Consists of 256 different colors
    random_colors     -> Randomly returns required(num_color) colors
    """
    colors = []
    random_colors = []
    # Generate 256 different colors and choose num_clors randomly
    for i in np.arange(0., 360., 360. / 256.):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))

    for i in range(0, num_colors):
        random_colors.append(colors[random.randint(0, len(colors) - 1)])
    return random_colors


def data_2d(n, r=None):
    data = []
    x = random.randint(-100, 100)
    y = random.randint(-100, 100)

    if r == None:
        r = random.uniform(0, 5)

    a = []
    b = []
    for i in range(n):
        a += [x + np.random.normal(scale=r)]
        b += [y + np.random.normal(scale=r)]

    return a, b


def build_data():
    data = []

    return data


def model():
    pass


def execute():
    with tf.Session() as sess:
        pass


if __name__ == '__main__':

    n_cluster = 30
    n = 100

    data_list = []
    for i in range(n_cluster):
        data_list += [data_2d(n, r=random.uniform(1, 30))]

    RGB_list = get_colors(n_cluster)
    for i in range(n_cluster):
        x, y = data_list[i]
        rgb = RGB_list[i]
        plot.scatter(x, y, c=rgb, s=1)
    plot.show()

    pass