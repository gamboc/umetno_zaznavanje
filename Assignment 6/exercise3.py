import numpy as np
import cv2
from matplotlib import pyplot as plt
import random
from a6_utils import *


def prepare_faces(path):
    faces = []
    face_size = (0, 0)

    for j in range(1, 65):
        number = str(j).zfill(3)
        name = path + number + ".png"

        image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
        face_size = image.shape
        image = np.reshape(image, -1)

        faces.append(image)

    return np.array(faces).T, face_size


def base_to_PCA(X, mean, U):
    if len(X.shape) == 1:
        X = np.expand_dims(X, -1)
    points = np.zeros((U.shape[1], X.shape[1]))

    for i, x in enumerate(X.T):
        print(i)
        points[:, i] = np.dot(U.T, x - mean)

    return points


def PCA_to_base(X, mean, U):
    if len(X.shape) == 1:
        X = np.expand_dims(X, -1)
    points = np.zeros((U.shape[0], X.shape[1]))

    for i, x in enumerate(X.T):
        points[:, i] = np.dot(U, x) + mean

    return points


def dualPCA(X):
    mean = np.mean(X, axis=1)
    X_d = (X.T - mean).T

    covariance_matrix = (1 / (X.shape[1] - 1)) * np.dot(X_d.T, X_d)
    U, S, VT = np.linalg.svd(covariance_matrix)
    S += 10e-15
    U = np.dot(X_d, U) * np.sqrt(1 / (S * X.shape[1]))

    return S, U, mean


def a():
    path = "data/faces/1/"
    # path = "data/faces/2/"
    # path = "data/faces/3/"

    print(prepare_faces(path)[0].shape)


def b():
    path = "data/faces/1/"
    #path = "data/faces/2/"
    # path = "data/faces/3/"

    faces, face_size = prepare_faces(path)
    S, U, mean = dualPCA(faces)

    test1 = np.copy(faces[:, 0])
    test1[4074] = 0

    test2 = faces[:, 0]
    test2_pca = base_to_PCA(test2, mean, U)
    test2_pca[0] = 0
    test2_2 = PCA_to_base(test2_pca, mean, U)

    for i in range(5):
        plt.subplot(2, 5, i + 1)
        vector = np.reshape(U[:, i], face_size)
        plt.imshow(vector, cmap='gray')

    plt.subplot(2, 4, 5)
    plt.imshow(np.reshape(faces[:, 0], face_size), cmap='gray')
    plt.subplot(2, 4, 6)
    plt.imshow(np.reshape(test1, face_size), cmap='gray')
    plt.subplot(2, 4, 7)
    plt.imshow(np.reshape(test2_2, face_size), cmap='gray')
    plt.subplot(2, 4, 8)
    plt.imshow(np.reshape(test2_2, face_size) - np.reshape(faces[:, 0], face_size), cmap='gray')

    print(np.where(np.reshape(test2_2, face_size) - np.reshape(faces[:, 0], face_size) == 0)[0])

    plt.show()


def c():
    path = "data/faces/1/"
    # path = "data/faces/2/"
    # path = "data/faces/3/"

    faces, face_size = prepare_faces(path)
    _, U, mean = dualPCA(faces)

    face = random.choice(faces.T)
    face_pca = base_to_PCA(face, mean, U)

    n = 32
    i = 1
    while n >= 1:
        plt.subplot(1, 6, i)
        face_pca[n:] = 0
        face_2 = PCA_to_base(face_pca, mean, U)
        plt.imshow(np.reshape(face_2, face_size), cmap='gray')
        plt.title(str(n))
        n //= 2
        i += 1
    plt.show()


# a()
b()
# ()