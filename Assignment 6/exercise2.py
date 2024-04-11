import numpy as np
from matplotlib import pyplot as plt
from a6_utils import *


def dualPCA_data(X):
    mean = np.mean(X, axis=1)
    X_d = (X.T - mean).T

    covariance_matrix = (1 / (X.shape[1] - 1)) * np.dot(X_d.T, X_d)
    U, S, VT = np.linalg.svd(covariance_matrix)
    U = np.dot(X_d, U) * np.sqrt(1 / (S * X.shape[1]))

    for i, vector in enumerate(U):
        U[i] = vector / np.linalg.norm(vector)

    return mean, covariance_matrix, S, U


def base_to_PCA(X, mean, U):
    if len(X.shape) == 1:
        X = np.expand_dims(X, -1)
    points = np.zeros((U.shape[1], X.shape[1]))

    for i, x in enumerate(X.T):
        points[:, i] = np.dot(U.T, x - mean)

    return points


def PCA_to_base(X, mean, U):
    if len(X.shape) == 1:
        X = np.expand_dims(X, -1)
    points = np.zeros((U.shape[0], X.shape[1]))

    for i, x in enumerate(X.T):
        points[:, i] = np.dot(U, x) + mean

    return points


def a():
    X = np.loadtxt("./data/points.txt").T

    mean, covariance_matrix, S, U = dualPCA_data(X)

    for i in range(len(S)):
        print(f"eigenvalue {i + 1}: {S[i]:.3f}, eigenvector: {U[:, i]} ({U[:, i]})")


def b():
    X = np.loadtxt("./data/points.txt").T
    mean, covariance_matrix, S, U = dualPCA_data(X)

    X_pca = base_to_PCA(X, mean, U)
    X_2 = PCA_to_base(X_pca, mean, U)

    for i in range(len(X.T)):
        distance = np.sqrt(np.sum((X.T[i] - X_2.T[i]) ** 2))
        print(f"error for point {i}: {distance}")

    plt.scatter(X[0], X[1], color='blue')
    plt.scatter(X_2[0], X_2[1], color='red')
    plt.show()


# a()
# b()