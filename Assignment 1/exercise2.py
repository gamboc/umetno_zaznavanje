from UZ_utils import *
import numpy as np
import cv2
from matplotlib import pyplot as plt


def grayscale(I):
    grayImage = np.zeros((len(I), len(I[0]), 1), dtype=np.float64)
    for i in range(len(grayImage)):
        for j in range(len(grayImage[0])):
            grayImage[i][j][0] = (I[i][j][0] + I[i][j][1] + I[i][j][2]) / 3

    return grayImage


def myhist(grayscale, bins):
    histogram = np.zeros(bins)
    reshaped = grayscale.reshape(-1)

    size_of_interval = 255 / bins
    for i in range(len(reshaped)):
        histogram[(reshaped[i] // size_of_interval).astype(np.uint8)] += 1

    return histogram


def a():
    bird = imread('./images/bird.jpg')
    gray = grayscale(bird)
    masked = np.copy(gray)
    masked[masked < 0.2] = 0
    masked[masked >= 0.2] = 1

    gray = np.where(gray < 0.25, 0, 1)

    plt.subplot(1, 3, 1)
    plt.imshow(bird)
    plt.subplot(1,3,2)
    plt.imshow(masked, cmap='gray')

    plt.subplot(1,3,3)
    plt.imshow(gray, cmap='gray')

    plt.show()


def b():
    bird = grayscale(imread('./images/bird.jpg'))
    bird = (bird * 255).astype(np.uint8)
    bins = 100
    histogram = myhist(bird, bins)
    indexes = range(1, bins + 1)
    plt.bar(indexes, histogram)
    plt.show()


#a()
b()