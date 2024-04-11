import numpy as np
import cv2
from matplotlib import pyplot as plt
import math


def gaussian(sigma):
    kernel_size = 2 * np.ceil(3 * sigma) + 1
    kernel = []

    for i in range(- int(kernel_size) // 2 + 1, int(kernel_size // 2) + 1):
        kernel.append(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- i**2 / (2 * sigma**2)))
    return np.array(kernel)


def gaussdx(sigma):
    kernel_size = 2 * np.ceil(3 * sigma) + 1
    kernel = []

    for i in range(-int(kernel_size) // 2 + 1, int(kernel_size // 2) + 1):
        kernel.append(- (1 / (np.sqrt(2 * np.pi) * sigma**3)) * i * np.exp(- (i**2 / (2*sigma**2))))

    kernel /= np.sum(np.abs(kernel))

    return np.array(kernel)


def partialdx(image, sigma):
    G = gaussian(sigma)[np.newaxis]
    kernel = np.array([-1, 1])[np.newaxis]
    return cv2.filter2D(cv2.filter2D(image, -1, G), -1, kernel)


def partialdy(image, sigma):
    G = gaussian(sigma)
    kernel = np.array([-1, 1])
    return cv2.filter2D(cv2.filter2D(image, -1, G), -1, kernel)


def gradient_magnitude(image, sigma):
    I_x = partialdx(image, sigma)
    I_y = partialdy(image, sigma)
    return np.sqrt(I_y**2 + I_x**2), np.arctan2(I_y, I_x)


def findedges(image, sigma, theta):
    mag, _ = gradient_magnitude(image, sigma)
    edges = np.where(mag >= theta, 1, 0)
    return edges


def a():
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY) / 255

    for i in range(1, 5):
        plt.subplot(1, 4, i)
        plt.imshow(findedges(museum, 1, i * 0.05), cmap='gray')

    plt.show()


def b():
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY) / 255
    plt.subplot(1, 3, 1)
    plt.imshow(museum, cmap='gray')

    mag, dir = gradient_magnitude(museum, 0.5)
    dir += np.pi
    mag[mag < 0.3] = 0

    meja_y, meja_x = mag.shape
    plt.subplot(1, 3, 2)
    plt.imshow(mag, cmap='gray')

    pi = np.pi
    for y in range(1, meja_y - 1):
        for x in range(1, meja_x - 1):
            if mag[y][x] != 0:
                if 0 <= dir[y][x] <= pi / 8 or 7 / 8 * pi < dir[y][x] <= 9 / 8 * pi or 15 / 8 * pi < dir[y][x] <= 2 * pi:
                    if mag[y][x - 1] < mag[y][x] and mag[y][x + 1] < mag[y][x]:
                        mag[y][x] = 1
                        mag[y][x - 1], mag[y][x + 1] = 0, 0
                    else:
                        mag[y][x] = 0
                elif pi / 8 <= dir[y][x] < 3 / 8 * pi or 9 / 8 * pi < 11 / 8 * pi:
                    if mag[y + 1][x + 1] < mag[y][x] and mag[y - 1][x - 1] < mag[y][x]:
                        mag[y][x] = 1
                        mag[y + 1][x + 1], mag[y - 1][x - 1] = 0, 0
                    else:
                        mag[y][x] = 0
                elif 3 / 8 * pi < dir[y][x] <= 5 / 8 * pi or 11 / 8 * pi < dir[y][x] <= 13 / 8 * pi:
                    if mag[y + 1][x] < mag[y][x] and mag[y - 1][x] < mag[y][x]:
                        mag[y][x] = 1
                        mag[y + 1][x], mag[y - 1][x] = 0, 0
                    else:
                        mag[y][x] = 0
                else:
                    if mag[y + 1][x - 1] < mag[y][x] and mag[y - 1][x + 1] < mag[y][x]:
                        mag[y][x] = 1
                        mag[y + 1][x - 1], mag[y - 1][x + 1] = 0, 0
                    else:
                        mag[y][x] = 0

    plt.subplot(1, 3, 3)
    plt.imshow(mag, cmap='gray')
    plt.show()

# a()
# b()
