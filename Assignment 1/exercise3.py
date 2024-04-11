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


def erode_then_dilate(image, SE):
    return cv2.dilate(cv2.erode(image, SE), SE)


def dilate_then_erode(image, SE):
    return cv2.erode(cv2.dilate(image, SE), SE)


def reverse(I, threshold):
    for i in range(len(I)):
        for j in range(len(I[i])):
            if I[i][j] < threshold:
                I[i][j] = 1
            else:
                I[i][j] = 0

    return I


def a():
    mask = imread('./images/mask.png')
    square = np.ones((3, 3))

    plt.subplot(2, 2, 1)
    plt.title("Eroding, then dilating (size = 3)")
    plt.imshow(erode_then_dilate(mask, square))

    plt.subplot(2, 2, 2)
    plt.title("Dilating, then eroding (size = 3)")
    plt.imshow(dilate_then_erode(mask, square))

    square = np.ones((5, 5))

    plt.subplot(2, 2, 3)
    plt.title("Eroding, then dilating (size = 5)")
    plt.imshow(erode_then_dilate(mask, square))

    plt.subplot(2, 2, 4)
    plt.title("Dilating, then eroding (size = 5)")
    plt.imshow(dilate_then_erode(mask, square))

    plt.show()


def b():
    bird = grayscale(imread('./images/bird.jpg'))
    bird[bird < 0.2] = 0
    bird[bird >= 0.2] = 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 20))
    bird = cv2.dilate(bird, ellipse)
    bird = cv2.erode(bird, ellipse)

    plt.imshow(bird, cmap='gray')
    plt.show()


def d():
    eagle = grayscale(imread('./images/eagle.jpg'))
    eagle[eagle < 0.6] = 0
    eagle[eagle >= 0.6] = 1
    plt.imshow(eagle, cmap='gray')
    plt.show()


def e():
    coins = imread('./images/coins.jpg')
    mask = grayscale(coins)
    plt.subplot(1, 2, 1)
    plt.imshow(mask, cmap='gray')

    plt.subplot(1, 2, 2)
    mask[mask < 0.9] = 0
    mask[mask >= 0.9] = 1
    mask = reverse(mask, 0.9)

    circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, circle)
    plt.imshow(mask, cmap='gray')
    plt.show()

    mask = (mask * 255).astype(np.uint8)
    number_of_components, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    print(number_of_components)
    print(stats)

    for i in range(1, len(stats)):
        if stats[i][-1] >= 700:
            for j in range(stats[i][0], stats[i][0] + stats[i][2]):
                for k in range(stats[i][1], stats[i][1] + stats[i][3]):
                    coins[k][j] = [1, 1, 1]
                    mask[k][j] = 1

    plt.subplot(1,2,1)
    plt.imshow(mask, cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(coins)
    plt.show()

#a()
#b()
#d()
e()