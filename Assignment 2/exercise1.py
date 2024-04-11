import numpy as np
import cv2
from matplotlib import pyplot as plt
import a2_utils


def simple_convolution(signal, kernel):
    result = []
    N = (len(kernel) - 1) // 2
    for i in range(len(signal) - N - 1):
        sum = 0
        for j in range(len(kernel)):
            if i + j >= len(signal):
                break
            sum += kernel[j]*signal[i + j]
        result.append(sum)

    return result


def b():
    signal = a2_utils.read_data("./signal.txt")
    kernel = a2_utils.read_data("./kernel.txt")
    result = simple_convolution(signal, kernel)

    plt.plot(signal, label="original")
    plt.plot(kernel, label="kernel")
    plt.plot(result, label="result")
    plt.plot(cv2.filter2D(signal, -1, kernel), label="cv2")
    plt.legend()

    plt.show()


def gauss(sigma):
    kernel_size = 2 * np.ceil(3 * sigma) + 1
    kernel = []

    for i in range(- int(kernel_size) // 2 + 1, int(kernel_size // 2) + 1):
        kernel.append(1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- i**2 / (2 * sigma**2)))
    return np.array(kernel)


def d():
    sigmas = [0.5, 1, 2, 3, 4]
    for sigma in sigmas:
        kernel = gauss(sigma)
        N = len(kernel) // 2
        print(kernel)
        plt.plot(range(-N, N + 1), kernel, label="sigma = " + str(sigma))
    plt.legend()

    plt.show()


def e():
    signal = a2_utils.read_data("./signal.txt")
    k1 = np.array(gauss(2))
    k2 = np.array([0.1, 0.6, 0.4])

    plt.subplot(1, 4, 1)
    plt.title("s")
    plt.plot(signal)

    plt.subplot(1, 4, 2)
    plt.title("(s * k1) * k2")
    plt.plot(cv2.filter2D(cv2.filter2D(signal, -1, k1), -1, k2))

    plt.subplot(1, 4, 3)
    plt.title("(s * k2) * k1")
    plt.plot(cv2.filter2D(cv2.filter2D(signal, -1, k2), -1, k1))

    plt.subplot(1, 4, 4)
    plt.title("s * (k1 * k2)")
    plt.plot(cv2.filter2D(signal, -1, cv2.filter2D(k1, -1, k2)))

    plt.show()


# b()
# d()
# e()
