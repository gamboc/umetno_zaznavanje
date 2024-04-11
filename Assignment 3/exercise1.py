import numpy as np
import cv2
from matplotlib import pyplot as plt


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


def c():
    impulse = np.zeros((50, 50))
    impulse[25, 25] = 1

    sigma = 4
    G, D = np.flip(gaussian(sigma))[np.newaxis], np.flip(gaussdx(sigma))[np.newaxis]
    plt.subplot(2, 3, 1)
    plt.title("Impulse")
    plt.imshow(impulse, cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("G, Dt")
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, G), -1, D.T), cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("D, Gt")
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, D), -1, G.T), cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("G, Gt")
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, G), -1, G.T), cmap='gray')

    plt.subplot(2, 3, 5)
    plt.title("Gt, D")
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, G.T), -1, D), cmap='gray')

    plt.subplot(2, 3, 6)
    plt.title("Dt, G")
    plt.imshow(cv2.filter2D(cv2.filter2D(impulse, -1, D.T), -1, G), cmap='gray')

    plt.show()


def d():
    museum = cv2.cvtColor(cv2.imread("./images/museum.jpg"), cv2.COLOR_BGR2GRAY) / 255
    museum_x = partialdx(museum, 1)
    museum_y = partialdy(museum, 1)

    museum_xx = partialdx(museum_x, 1)
    museum_xy = partialdy(museum_x, 1)
    museum_yy = partialdy(museum_y, 1)

    museum_mag, museum_dir = gradient_magnitude(museum, 1)

    plt.subplot(2, 4, 1)
    plt.title("Original")
    plt.imshow(museum, cmap='gray')

    plt.subplot(2, 4, 2)
    plt.title("I_x")
    plt.imshow(museum_x, cmap='gray')

    plt.subplot(2, 4, 3)
    plt.title("I_y")
    plt.imshow(museum_y, cmap='gray')

    plt.subplot(2, 4, 4)
    plt.title("I_mag")
    plt.imshow(museum_mag, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.title("I_xx")
    plt.imshow(museum_xx, cmap='gray')

    plt.subplot(2, 4, 6)
    plt.title("I_xy")
    plt.imshow(museum_xy, cmap='gray')

    plt.subplot(2, 4, 7)
    plt.title("I_yy")
    plt.imshow(museum_yy, cmap='gray')

    plt.subplot(2, 4, 8)
    plt.title("I_dir")
    plt.imshow(museum_dir, cmap='gray')

    plt.show()


# c()
# d()