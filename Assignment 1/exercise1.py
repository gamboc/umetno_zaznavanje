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


def a():
    I = imread('./images/umbrellas.jpg')
    print(I.shape)
    print(I[0][0])
    I2 = (255*I).astype(np.uint8)
    print(I2[0][0])
    imshow(I2)


def b():
    I = imread('./images/umbrellas.jpg')
    gray = grayscale(I)
    plt.imshow(gray, cmap='gray')
    plt.show()


def c():
    gray = grayscale(imread('./images/umbrellas.jpg'))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    cutout = gray[100:200, 200:400]
    plt.subplot(1, 2, 2)
    plt.imshow(cutout, cmap='gray')
    plt.show()


def d(low_x, high_x, low_y, high_y):
    I = imread('./images/umbrellas.jpg')
    inverted_rectangle = np.copy(I)
    inverted_rectangle[low_x:high_x, low_y:high_y] = 1 - I[low_x:high_x, low_y:high_y]
    imshow(inverted_rectangle)


def e():
    I = imread('./images/umbrellas.jpg')
    plt.subplot(1,2,1)
    plt.imshow(I)
    reducedUmbrellas = grayscale(I) * 0.3
    plt.subplot(1,2,2)
    plt.imshow(reducedUmbrellas, vmax=1, cmap='gray')
    plt.show()


#a()
#b()
#c()
#d(130, 260, 200, 400)
#e()