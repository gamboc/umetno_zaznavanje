import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils


def triangulate(correspondences, P1, P2):
    result = []

    for p1_x, p1_y, p2_x, p2_y in correspondences:
        points1 = np.array([[0, -1, p1_y], [1, 0, -p1_x], [-p1_y, p1_x, 0]])
        points2 = np.array([[0, -1, p2_y], [1, 0, -p2_x], [-p2_y, p2_x, 0]])

        A1 = points1 @ P1
        A2 = points2 @ P2

        A = [A1[0], A1[1], A2[0], A2[1]]

        U, S, VT = np.linalg.svd(A)

        X = VT[-1]
        X /= X[-1]

        result.append(X[:3])

    return result


def a():
    file = open("data/epipolar/house_points.txt", "r")
    points = np.array(file.read().split()).reshape((-1, 4)).astype(np.float64)
    points1 = points[:, 0:2]
    points2 = points[:, 2:4]

    house1 = cv2.cvtColor(cv2.imread('data/epipolar/house1.jpg'), cv2.COLOR_BGR2GRAY).astype(np.float64)
    house2 = cv2.cvtColor(cv2.imread('data/epipolar/house2.jpg'), cv2.COLOR_BGR2GRAY).astype(np.float64)
    plt.subplot(1, 2, 1)
    plt.imshow(house1, cmap='gray')
    plt.scatter(points1[:, 0], points1[:, 1], color='red')
    for i, (x, y) in enumerate(points1):
        plt.text(x, y, i)

    plt.subplot(1, 2, 2)
    plt.imshow(house2, cmap='gray')
    plt.scatter(points2[:, 0], points2[:, 1], color='red')
    for i, (x, y) in enumerate(points2):
        plt.text(x, y, i)

    plt.show()

    file = open("data/epipolar/house1_camera.txt", "r")
    camera1 = np.array(file.read().split()).reshape((3, 4)).astype(np.float64)
    file = open("data/epipolar/house2_camera.txt", "r")
    camera2 = np.array(file.read().split()).reshape((3, 4)).astype(np.float64)

    points_3d = triangulate(points, camera1, camera2)
    T = np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])
    points_3d = np.dot(points_3d, T)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, (x, y, z) in enumerate(points_3d):
        ax.plot([x], [y], [z], 'ro')
        ax.text(x, y, z, str(i))

    plt.show()


a()
