import numpy as np
import cv2
from matplotlib import pyplot as plt


def gauss(sigma):
    kernel_size = 2 * np.ceil(3 * sigma) + 1
    kernel = []

    for i in range(- int(kernel_size) // 2 + 1, int(kernel_size // 2) + 1):
        kernel.append(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- i**2 / (2 * sigma**2)))
    return np.array(kernel)


def partialdx(image, sigma):
    G = gauss(sigma)[np.newaxis]
    kernel = np.array([-1, 1])[np.newaxis]
    return cv2.filter2D(cv2.filter2D(image, -1, G), -1, kernel)


def partialdy(image, sigma):
    G = gauss(sigma)
    kernel = np.array([-1, 1])
    return cv2.filter2D(cv2.filter2D(image, -1, G), -1, kernel)


def gradient_magnitude(image, sigma):
    I_x = partialdx(image, sigma)
    I_y = partialdy(image, sigma)
    return np.arctan2(I_y, I_x), np.sqrt(I_y**2 + I_x**2)


def findedges(image, sigma, theta):
    _, mag = gradient_magnitude(image, sigma)

    mag[mag >= theta] = 1
    mag[mag < theta] = 0

    return mag


def a():
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY) / 255

    for i in range(1, 5):
        plt.subplot(1, 4, i)
        plt.imshow(findedges(museum, 1, i * 0.05), cmap='gray')

    plt.show()


def b():
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY) / 255
    museum[museum <= 0.5] = 0
    museum = cv2.filter2D(museum, -1, gauss(0.5))

    museum = findedges(museum, 0.5, 0.16)

    plt.subplot(2, 2, 1)
    plt.title("thresholded")
    plt.imshow(museum, cmap='gray')

    dir, mag = gradient_magnitude(museum, 1)

    plt.subplot(2, 2, 2)
    plt.title("mag")
    plt.imshow(mag, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title("dir")
    plt.imshow(dir, cmap='gray')

    y_len, x_len = mag.shape
    pi = np.pi

    for y in range(1, y_len - 1):
        for x in range(1, x_len - 1):
            n1, n2 = 1, 1
            if (-pi / 8 <= dir[y][x] < pi / 8) or (-pi <= dir[y][x] < -7 * pi / 8) or (7 * pi / 8 <= dir[y][x] <= pi):
                n1 = museum[y][x + 1]
                n2 = museum[y][x - 1]
            elif (pi / 8 <= dir[y][x] < 3 * pi / 8) or (-7 * pi / 8 <= dir[y][x] < -5 * pi / 8):
                n1 = museum[y + 1][x - 1]
                n2 = museum[y - 1][x + 1]
            elif (3 * pi / 8 <= dir[y][x] < 5 * pi / 8) or (-5 * pi / 8 <= dir[y][x] < -3 * pi / 8):
                n1 = museum[y + 1][x]
                n2 = museum[y - 1][x]
            elif (5 * pi / 8 <= dir[y][x] < 7 * pi / 8) or (-3 * pi / 8 <= dir[y][x] < -pi / 8):
                n1 = museum[y - 1][x - 1]
                n2 = museum[y + 1][x + 1]

            if museum[y][x] > n1 and museum[y][x] > n2:
                museum[y][x] = 1

    plt.subplot(2, 2, 4)
    plt.title("result")
    plt.imshow(museum, cmap='gray')
    plt.show()

#a()
b()
