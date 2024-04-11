import numpy as np
import cv2
from matplotlib import pyplot as plt

from a4_utils import convolve


def gauss(sigma):
    N = np.ceil(3 * sigma)
    xs = np.arange(-N, N+1)

    kernel = (1/(2*np.pi*sigma)**0.5) * np.exp((-1) * xs**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)

    return kernel


def gaussdx(sigma):
    N = np.ceil(3 * sigma)
    xs = np.arange(-N, N+1)

    kernel = (-1 / (((2 * np.pi) ** 0.5) * (sigma ** 3))) * xs * np.exp(((-1) * (xs ** 2)) / (2 * (sigma ** 2)))
    kernel /= np.sum(np.abs(kernel))

    return kernel

def get_coordinates(image):
    return (image > 0).nonzero()


def reformat_points(points, ix):
    out = np.zeros((2, len(ix)))
    out[0] = [points[0][i] for i in ix]
    out[1] = [points[1][i] for i in ix]
    return np.fliplr(out.T)


def nonmaxima_suppression(H, threshold, box_size):
    out = np.zeros(H.shape)
    H = np.vstack((np.zeros((box_size, H.shape[1])), H, np.zeros((box_size, H.shape[1]))))
    H = np.hstack((np.zeros((H.shape[0], box_size)), H, np.zeros((H.shape[0], box_size))))

    for y in range(len(out)):
        for x in range(len(out[y])):
            neighborhood = H[y:y+2*box_size + 1, x:x+2*box_size + 1]
            if neighborhood[box_size, box_size] == np.max(neighborhood) and neighborhood[box_size, box_size] > threshold:
                out[y, x] = 1

    return out


def hessian_points(image, sigma, threshold, box_size, nonmaxima = False):
    G = gauss(sigma)
    N = int((len(G) - 1) / 2)
    G = np.vstack((np.zeros((N, len(G))), G, np.zeros((N, len(G)))))

    D = gaussdx(sigma)[::-1]
    N = int((len(D) - 1) / 2)
    D = np.vstack((np.zeros((N, len(D))), D, np.zeros((N, len(D)))))

    image_x = convolve(convolve(image, G.T), D)
    image_y = convolve(convolve(image, D.T), G)
    image_xx = convolve(convolve(image_x, G.T), D)
    image_xy = convolve(convolve(image_x, D.T), G)
    image_yy = convolve(convolve(image_y, D.T), G)

    det_H = image_xx*image_yy - image_xy**2

    if nonmaxima:
        return nonmaxima_suppression(det_H, threshold, box_size)
    else:
        return det_H


def harris_points(image, sigma, sigma2, alpha, threshold, box_size, nonmaxima = False):
    G = gauss(sigma)
    N = int((len(G) - 1) / 2)
    G = np.vstack((np.zeros((N, len(G))), G, np.zeros((N, len(G)))))

    D = gaussdx(sigma)[::-1]
    N = int((len(D) - 1) / 2)
    D = np.vstack((np.zeros((N, len(D))), D, np.zeros((N, len(D)))))

    image_x = convolve(convolve(image, G.T), D)
    image_y = convolve(convolve(image, D.T), G)

    G = gauss(sigma2)
    N = int((len(G) - 1) / 2)
    G = np.vstack((np.zeros((N, len(G))), G, np.zeros((N, len(G)))))

    Gimage_xy = convolve(convolve(image_x * image_y, G), G.T)
    Gimage_xx = convolve(convolve(image_x ** 2, G), G.T)
    Gimage_yy = convolve(convolve(image_y ** 2, G), G.T)

    det = Gimage_xx * Gimage_yy - Gimage_xy**2
    trace = Gimage_xx + Gimage_yy

    if nonmaxima:
        return nonmaxima_suppression(det - alpha * (trace**2), threshold, box_size)
    else:
        return det - alpha * (trace**2)


def a():
    graf = cv2.cvtColor(cv2.imread("./data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
    sigmas = [3, 6, 9]
    threshold = 0.004

    for i in range(len(sigmas)):
        plt.subplot(2, 3, i + 1)
        plt.title("sigma = " + str(sigmas[i]))
        hessian = hessian_points(graf, sigmas[i], threshold, 1)
        plt.imshow(hessian)

        plt.subplot(2, 3, 3 + i + 1)
        plt.imshow(graf, cmap='gray')
        suppressed = nonmaxima_suppression(hessian, threshold, 1)
        suppressed = (suppressed > 0).nonzero()
        plt.scatter(suppressed[1], suppressed[0], marker='x', color='red')

    plt.show()


def b():
    graf = cv2.cvtColor(cv2.imread("./data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
    sigmas = [3, 6, 9]
    threshold = 1e-6

    for i in range(len(sigmas)):
        plt.subplot(2, 3, i + 1)
        plt.title("sigma = " + str(sigmas[i]))
        harris = harris_points(graf, sigmas[i], 1.6 * sigmas[i], 0.06, threshold, 1)
        plt.imshow(harris)

        plt.subplot(2, 3, 3 + i + 1)
        plt.imshow(graf, cmap='gray')
        suppressed = nonmaxima_suppression(harris, threshold, 1)
        suppressed = (suppressed > 0).nonzero()
        plt.scatter(suppressed[1], suppressed[0], marker='x', color='red')

    plt.show()


# a()
# b()