
import numpy as np
import math


def m(x, y, f):
    """Modulating function"""
    return np.cos(2 * np.pi * f * math.sqrt(x ** 2 + y ** 2))

def gabor(x, y, dx, dy, f):
    """Spatial filter"""
    return (1 / (2 * math.pi * dx * dy)) * np.exp(-0.5 * (x ** 2 / dx ** 2 + y ** 2 / dy ** 2)) * m(x, y, f)

def spatial(f, dx, dy):
    """Calculates spatial filter over 8x8 blocks."""
    sfilter = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            sfilter[i, j] = gabor((-4 + j), (-4 + i), dx, dy, f)
    return sfilter

def get_vec(convolvedtrain1, convolvedtrain2):
    """Gets feature vector from convolution outputs."""
    feature_vec = []
    for i in range(6):
        for j in range(64):
            start_height = i * 8
            end_height = start_height + 8
            start_wid = j * 8
            end_wid = start_wid + 8
            grid1 = convolvedtrain1[start_height:end_height, start_wid:end_wid]
            grid2 = convolvedtrain2[start_height:end_height, start_wid:end_wid]

            # Channel 1
            absolute = np.absolute(grid1)
            mean = np.mean(absolute)
            feature_vec.append(mean)
            std = np.mean(np.abs(absolute - mean))
            feature_vec.append(std)

            # Channel 2
            absolute = np.absolute(grid2)
            mean = np.mean(absolute)
            feature_vec.append(mean)
            std = np.mean(np.abs(absolute - mean))
            feature_vec.append(std)

    return feature_vec
