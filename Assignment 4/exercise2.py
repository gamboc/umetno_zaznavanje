import numpy as np
import cv2
import os
import math
import sys
import random
from matplotlib import pyplot as plt

from a4_utils import *


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


def harris_points(image, sigma, sigma2, alpha, threshold, box_size, nonmaxima=False):
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


def find_correspondences(descriptors_a, descriptors_b):
    a = range(len(descriptors_a))
    b = np.zeros(len(descriptors_a), dtype = int)
    for i, descriptor in enumerate(descriptors_a):
        distances = ((0.5 * np.sum((descriptors_b**0.5 - descriptor**0.5) ** 2, axis=1))**0.5)
        b[i] = np.where(distances == np.min(distances))[0][0]
    return a, b


def find_matches(image_a, image_b, sigma, alpha, threshold, box_size):
    points_a = get_coordinates(harris_points(image_a, sigma, 1.6*sigma, alpha, threshold, box_size, nonmaxima=True))
    points_b = get_coordinates(harris_points(image_b, sigma, 1.6*sigma, alpha, threshold, box_size, nonmaxima=True))

    descriptors_a = simple_descriptors(image_a, points_a[0], points_a[1])
    descriptors_b = simple_descriptors(image_b, points_b[0], points_b[1])

    correspondences_1 = set((tuple(i) for i in np.array(find_correspondences(descriptors_a, descriptors_b)).T))
    correspondences_2 = set((tuple(i) for i in np.fliplr(np.array(find_correspondences(descriptors_b, descriptors_a)).T)))

    correspondences = correspondences_1.intersection(correspondences_2)

    return np.array(list(correspondences))


def a():
    graf_a = cv2.cvtColor(cv2.imread("./data/graf/graf_a_small.jpg"), cv2.COLOR_BGR2GRAY) / 255
    graf_b = cv2.cvtColor(cv2.imread("./data/graf/graf_b_small.jpg"), cv2.COLOR_BGR2GRAY) / 255

    points1 = get_coordinates(harris_points(graf_a, 3, 1.6 * 3, 0.06, 1e-6, 10, nonmaxima=True))
    points2 = get_coordinates(harris_points(graf_b, 3, 1.6 * 3, 0.06, 1e-6, 10, nonmaxima=True))

    descriptors_a = simple_descriptors(graf_a, points1[0], points1[1])
    descriptors_b = simple_descriptors(graf_b, points2[0], points2[1])
    a, b = find_correspondences(descriptors_a, descriptors_b)

    display_matches(graf_a, reformat_points(points1, a), graf_b, reformat_points(points2, b))


def b():
    graf_a = cv2.cvtColor(cv2.imread("./data/graf/graf_a_small.jpg"), cv2.COLOR_BGR2GRAY) / 255
    graf_b = cv2.cvtColor(cv2.imread("./data/graf/graf_b_small.jpg"), cv2.COLOR_BGR2GRAY) / 255

    points1 = get_coordinates(harris_points(graf_a, 3, 1.6 * 3, 0.06, 1e-6, 10, nonmaxima=True))
    points2 = get_coordinates(harris_points(graf_b, 3, 1.6 * 3, 0.06, 1e-6, 10, nonmaxima=True))

    matches = find_matches(graf_a, graf_b, 3, 0.06, 1e-6, 10)

    display_matches(graf_a, reformat_points(points1, matches[::, 0]), graf_b, reformat_points(points2, matches[::, 1]))


a()
b()
