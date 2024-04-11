import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils


def fundamental_matrix(points1, points2):
    points1, T1 = a5_utils.normalize_points(points1)
    points2, T2 = a5_utils.normalize_points(points2)

    u = points2[:, 0]
    v = points2[:, 1]
    u_p = points1[:, 0]
    v_p = points1[:, 1]

    A = np.stack([u*u_p, u*v_p, u, v*u_p, v*v_p, v, u_p, v_p, np.ones(u.shape)]).T
    U, S, VT = np.linalg.svd(A)
    Ft = VT.T[:, -1].reshape((3, 3))

    U, S, VT = np.linalg.svd(Ft)
    S[-1] = 0
    F = (U * S) @ VT
    F = T2.T @ F @ T1

    return F


def b():
    file = open("data/epipolar/house_points.txt", "r")
    points = np.array(file.read().split()).reshape((-1, 4)).astype(np.float64)
    points1 = points[:, 0:2]
    points2 = points[:, 2:4]

    F = fundamental_matrix(points1, points2)

    house1 = cv2.cvtColor(cv2.imread('data/epipolar/house1.jpg'), cv2.COLOR_BGR2GRAY).astype(np.float64)
    house2 = cv2.cvtColor(cv2.imread('data/epipolar/house2.jpg'), cv2.COLOR_BGR2GRAY).astype(np.float64)

    plt.subplot(1, 2, 1)
    plt.imshow(house1, cmap='gray')
    for point in points2:
        p_hom = np.array([point[0], point[1], 1])
        line = F.T @ p_hom
        a5_utils.draw_epiline(line, house1.shape[0], house1.shape[1])
    plt.scatter(points1[:, 0], points1[:, 1], color='red')

    plt.subplot(1, 2, 2)
    plt.imshow(house2, cmap='gray')
    for point in points1:
        p_hom = np.array([point[0], point[1], 1])
        line = F @ p_hom
        a5_utils.draw_epiline(line, house2.shape[0], house1.shape[1])
    plt.scatter(points2[:, 0], points2[:, 1], color='red')

    plt.show()


def reprojection_error(x1, x2, F):
    p1_hom = np.array([x1[0], x1[1], 1])
    p2_hom = np.array([x2[0], x2[1], 1])

    line1 = F.T @ p2_hom
    line2 = F @ p1_hom

    distance1 = np.abs(line1[0]*x1[0] + line1[1]*x1[1] + line1[2]) / np.sqrt(line1[0]**2 + line1[1]**2)
    distance2 = np.abs(line2[0]*x2[0] + line2[1]*x2[1] + line2[2]) / np.sqrt(line2[0] ** 2 + line2[1] ** 2)

    return (distance1 + distance2) / 2


def c():
    file = open("data/epipolar/house_points.txt", "r")
    points = np.array(file.read().split()).reshape((-1, 4)).astype(np.float64)
    points1 = points[:, 0:2]
    points2 = points[:, 2:4]

    F = fundamental_matrix(points1, points2)

    print("Estimated error: 0.15")
    print(reprojection_error(np.array([85, 233]), np.array([67, 219]), F))

    amount = 0
    error_sum = 0
    for x1, x2 in zip(points1, points2):
        error_sum += reprojection_error(x1, x2, F)
        amount += 1

    print("Estimated error: 0.33")
    print(error_sum/amount)


b()
c()