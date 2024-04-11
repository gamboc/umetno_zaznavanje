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


def hessian_points(image, sigma, threshold, box_size, nonmaxima=False):
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


def find_matches(image_a, image_b, sigma, alpha, threshold, box_size):
    points_a = get_coordinates(harris_points(image_a, sigma, 1.6*sigma, alpha, threshold, box_size, nonmaxima=True))
    points_b = get_coordinates(harris_points(image_b, sigma, 1.6*sigma, alpha, threshold, box_size, nonmaxima=True))

    descriptors_a = simple_descriptors(image_a, points_a[0], points_a[1])
    descriptors_b = simple_descriptors(image_b, points_b[0], points_b[1])

    correspondences_1 = set((tuple(i) for i in np.array(find_correspondences(descriptors_a, descriptors_b)).T))
    correspondences_2 = set((tuple(i) for i in np.fliplr(np.array(find_correspondences(descriptors_b, descriptors_a)).T)))

    correspondences = correspondences_1.intersection(correspondences_2)

    return np.array(list(correspondences))


def find_correspondences(descriptors_a, descriptors_b):
    a = range(len(descriptors_a))
    b = np.zeros(len(descriptors_a), dtype = int)
    for i, descriptor in enumerate(descriptors_a):
        distances = ((0.5 * np.sum((descriptors_b**0.5 - descriptor**0.5) ** 2, axis=1))**0.5)
        b[i] = np.where(distances == np.min(distances))[0][0]
    return a, b


def estimate_homography(correspondences):
    points_a, points_b = np.array_split(correspondences, 2, axis=1)
    A = np.zeros((2 * len(points_a), 9))

    for i in range(len(points_a)):
        A[2 * i] = [points_a[i][0], points_a[i][1], 1, 0, 0, 0, -points_b[i][0] * points_a[i][0], -points_b[i][0] * -points_a[i][1], -points_b[i][0]]
        A[2 * i + 1] = [0, 0, 0, points_a[i][0], points_a[i][1], 1, -points_b[i][1] * points_a[i][0], -points_b[i][1] * -points_a[i][1], -points_b[i][1]]

    _, _, VT = np.linalg.svd(A)
    line = VT[-1]
    line = line / line[-1]

    return np.reshape(line, (3, 3))


def a():
    newyork = True

    if newyork:
        image_a = cv2.cvtColor(cv2.imread("./data/newyork/newyork_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
        image_b = cv2.cvtColor(cv2.imread("./data/newyork/newyork_b.jpg"), cv2.COLOR_BGR2GRAY) / 255
        correspondences = np.loadtxt("./data/newyork/newyork.txt", dtype=int)
    else:
        image_a = cv2.cvtColor(cv2.imread("./data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
        image_b = cv2.cvtColor(cv2.imread("./data/graf/graf_b.jpg"), cv2.COLOR_BGR2GRAY) / 255
        correspondences = np.loadtxt("./data/graf/graf.txt", dtype=int)

    points_a, points_b = np.array_split(correspondences, 2, axis=1)
    H = estimate_homography(correspondences)
    display_matches(image_a, points_a, image_b, points_b)

    warped_a = cv2.warpPerspective(image_a, H, image_a.shape[::-1])

    plt.subplot(1, 3, 1)
    plt.title("Original image_a")
    plt.imshow(image_a, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Warped image_a")
    plt.imshow(np.where(warped_a == 0, image_b / 2, warped_a), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Original image_b")
    plt.imshow(image_b, cmap='gray')

    plt.show()
    print(H)


def b():
    newyork = False
    sigma = 3
    sigma2 = 1.6
    alpha = 0.06
    box_size = 10

    if newyork:
        image_a = cv2.cvtColor(cv2.imread("./data/newyork/newyork_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
        image_b = cv2.cvtColor(cv2.imread("./data/newyork/newyork_b.jpg"), cv2.COLOR_BGR2GRAY) / 255
    else:
        image_a = cv2.cvtColor(cv2.imread("./data/graf/graf_a.jpg"), cv2.COLOR_BGR2GRAY) / 255
        image_b = cv2.cvtColor(cv2.imread("./data/graf/graf_b.jpg"), cv2.COLOR_BGR2GRAY) / 255

    threshold = 1e-6
    points_a = get_coordinates(harris_points(image_a, sigma, sigma2 * sigma, alpha, threshold, box_size, nonmaxima=True))
    points_b = get_coordinates(harris_points(image_b, sigma, sigma2 * sigma, alpha, threshold, box_size, nonmaxima=True))

    matches = find_matches(image_a, image_b, sigma, alpha, threshold, box_size)
    points_a = reformat_points(points_a, matches[::, 0])
    points_b = reformat_points(points_b, matches[::, 1])
    display_matches(image_a, points_a, image_b, points_b)

    min_error = float('inf')
    best_H = np.identity(3)
    for i in range(30):
        picks = random.sample(range(len(matches)), 4)
        H = estimate_homography(np.hstack((points_a[picks], points_b[picks])).astype(int))

        reprojections = H @ np.vstack((points_a.T, np.ones(points_a.shape[0])))
        reprojections /= reprojections[2]
        reprojections = reprojections[:2].T

        error = np.sum((reprojections - points_b)**2, axis=1)**0.5
        inlier_mask = np.where(error < 25, True, False) # 25 is the threshold
        error = np.average(error)

        if error < min_error:
            min_error = error
            best_H = H

        if np.sum(inlier_mask) / points_a.shape[0] > 0.7:
            H = estimate_homography(np.hstack((points_a[inlier_mask], points_b[inlier_mask])).astype(int))
            reprojections = H @ np.vstack((points_a.T, np.ones(points_b.shape[0])))
            reprojections /= reprojections[2]
            reprojections = reprojections[:2].T

            error = np.sum((reprojections - points_b) ** 2, axis=1) ** 0.5
            error = np.average(error)
            if error < min_error:
                min_error = error
                best_H = H

    warped_a = cv2.warpPerspective(image_a, best_H, image_a.shape[::-1])
    plt.subplot(1, 3, 1)
    plt.title("Original image_a")
    plt.imshow(image_a, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Warped image_a")
    plt.imshow(np.where(warped_a == 0, image_b / 2, warped_a), cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Original image_b")
    plt.imshow(image_b, cmap='gray')

    plt.show()


# a()
b()