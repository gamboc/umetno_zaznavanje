import numpy as np
import cv2
from matplotlib import pyplot as plt
import a2_utils
import exercise1


def gaussfilter(image, sigma):
    kernel = exercise1.gauss(sigma)
    image = cv2.filter2D(image, -1, kernel)
    image = cv2.filter2D(image, -1, kernel.T)

    return image


def sp(signal, percent):
    result = signal.copy()

    result[np.random.rand(signal.shape[0]) < percent / 2] = 0
    result[np.random.rand(signal.shape[0]) < percent / 2] = 4

    return result


def simple_median(signal, w):
    for i in range(0, len(signal) - w + 1, 2):
        temp = signal[i:(i+w)]
        temp.sort()
        if w % 2 == 1:
            for j in range(i, len(temp) + i):
                signal[j] = temp[w // 2]
        else:
            for j in range(i, len(temp) + i):
                signal[j] = (temp[w // 2] + temp[w // 2 + 1]) / 2

    return signal


def a():
    lena = (cv2.cvtColor(cv2.imread("./images/lena.png"), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255)
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(lena, cmap='gray')

    plt.subplot(2, 3, 2)
    lena_gauss = a2_utils.gauss_noise(lena, 0.105)
    plt.title("Gauss noise")
    plt.imshow(lena_gauss, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("Salt and pepper noise")
    lena_sp = a2_utils.sp_noise(lena, 0.1)
    plt.imshow(lena_sp, cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Gauss noise, gauss filter")
    plt.imshow(gaussfilter(lena_gauss, 1), cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("Salt and pepper noise, gauss filter")
    plt.imshow(gaussfilter(lena_sp, 1), cmap='gray')
    plt.show()


def b():
    kernel = np.array([[0, 0, 0], [0, 2, 0], [0, 0, 0]]) - 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY)
    museum_sharp = cv2.filter2D(museum, -1, kernel)

    plt.subplot(1, 2, 1)
    plt.imshow(museum, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(museum_sharp, cmap='gray')

    plt.show()


def c():
    signal = np.zeros(100)
    signal[20:50] = 1
    plt.subplot(1, 4, 1)
    plt.title("Original")
    plt.plot(signal)

    signal_sp = sp(signal, 0.1)
    plt.subplot(1, 4, 2)
    plt.title("Salt and pepper noise")
    plt.plot(signal_sp)

    signal_gauss = gaussfilter(signal_sp, 3)
    plt.subplot(1, 4, 3)
    plt.title("Gauss filter")
    plt.plot(signal_gauss)

    signal_median = simple_median(signal, 3)
    plt.subplot(1, 4, 4)
    plt.title("Median")
    plt.plot(signal_median)

    plt.show()


# a()
# b()
# c()
