import numpy as np
from matplotlib import pyplot as plt
from a6_utils import *

def PCA_data(X):
    mean = np.mean(X, axis = 1)
    X_d = (X.T - mean).T

    covariance_matrix = (1 / (X.shape[1] - 1)) * np.dot(X_d, X_d.T)

    U, S, VT = np.linalg.svd(covariance_matrix)
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


def index_of_closest(array, point):
    idx = 0
    smallest_distance = float('inf')
    for i, p in enumerate(array.T):
        distance = np.sum((p - point) ** 2)
        if distance < smallest_distance:
            idx = i
            smallest_distance = distance

    return idx


def a():
    X = np.array([[3, 3, 7, 6], [4, 6, 6, 4]])
    _, _, S, U = PCA_data(X)
    print(f"first eigenvector: {U[:, 0]}")
    print(f"second eigenvector: {U[:, 1]}")
    print(f"first eigenvalue:  {S[0]}")
    print(f"second eigenvalue: {S[1]}")


def b():
    X = np.loadtxt("./data/points.txt", dtype = int).T
    print(X)
    mean, covariance_matrix, S, U = PCA_data(X)

    plt.scatter(X[0], X[1], marker="x", color='blue')
    drawEllipse(mean, covariance_matrix)

    plt.show()


def c():
    X = np.loadtxt("./data/points.txt", dtype = int).T
    mean, covariance_matrix, S, U = PCA_data(X)

    plt.scatter(X[0], X[1], marker="x", color='blue')
    plt.scatter(mean[0], mean[1], marker="x", color='red')
    drawEllipse(mean, covariance_matrix)

    vector1 = U[:, 0] * np.sqrt(S[0])
    vector2 = U[:, 1] * np.sqrt(S[1])

    plt.plot([mean[0], mean[0] + vector1[0]], [mean[1], mean[1] + vector1[1]], color='red')
    plt.plot([mean[0], mean[0] + vector2[0]], [mean[1], mean[1] + vector2[1]], color='green')
    plt.show()


def d():
    X = np.loadtxt("./data/points.txt", dtype=int).T
    mean, covariance_matrix, S, U = PCA_data(X)

    cumulative_eigenvalues = np.zeros(len(S))
    for i in range(len(S)):
        cumulative_eigenvalues[i:] += S[i]

    cumulative_eigenvalues /= cumulative_eigenvalues[-1]

    plt.bar(range(len(cumulative_eigenvalues)), cumulative_eigenvalues)
    print(cumulative_eigenvalues[0])

    plt.show()


def e():
    X = np.loadtxt("./data/points.txt", dtype=int).T
    mean, covariance_matrix, S, U = PCA_data(X)
    X_pca = base_to_PCA(X, mean, U)

    U_d = np.copy(U)
    U_d[:, 1] = 0
    X_2 = PCA_to_base(X_pca, mean, U_d)

    plt.subplot(1, 2, 1)
    plt.scatter(X[0], X[1])
    drawEllipse(mean, covariance_matrix)

    vector1 = U[:, 0] * np.sqrt(S[0])
    vector2 = U[:, 1] * np.sqrt(S[1])

    plt.plot([mean[0], mean[0] + vector1[0]], [mean[1], mean[1] + vector1[1]], color='red')
    plt.plot([mean[0], mean[0] + vector2[0]], [mean[1], mean[1] + vector2[1]], color='green')

    plt.subplot(1, 2, 2)
    plt.scatter(X_2[0], X_2[1])
    drawEllipse(mean, covariance_matrix)

    vector1 = U[:, 0] * np.sqrt(S[0])
    vector2 = U[:, 1] * np.sqrt(S[1])

    plt.plot([mean[0], mean[0] + vector1[0]], [mean[1], mean[1] + vector1[1]], color='red')
    plt.plot([mean[0], mean[0] + vector2[0]], [mean[1], mean[1] + vector2[1]], color='green')

    plt.show()


def f():
    X = np.loadtxt("./data/points.txt", dtype=int).T
    point = np.array([6, 6])
    closest = index_of_closest(X, point)
    print(closest)

    mean, covariance_matrix, S, U = PCA_data(X)
    X_pca = base_to_PCA(X, mean, U)
    point_pca = base_to_PCA(point, mean, U)

    U_d = np.copy(U)
    U_d[:, 1] = 0

    X_2 = PCA_to_base(X_pca, mean, U_d)
    point_2 = PCA_to_base(point_pca, mean, U_d)
    closest = index_of_closest(X_pca, point_pca)
    print(closest)

    plt.subplot(1, 2, 1)
    plt.scatter(point[0], point[1], color='orange')
    plt.scatter(X[0], X[1])
    drawEllipse(mean, covariance_matrix)

    vector1 = U[:, 0] * np.sqrt(S[0])
    vector2 = U[:, 1] * np.sqrt(S[1])

    plt.plot([mean[0], mean[0] + vector1[0]], [mean[1], mean[1] + vector1[1]], color='red')
    plt.plot([mean[0], mean[0] + vector2[0]], [mean[1], mean[1] + vector2[1]], color='green')

    plt.subplot(1, 2, 2)
    plt.scatter(point_2[0], point_2[1], color='orange')
    plt.scatter(X_2[0], X_2[1])
    drawEllipse(mean, covariance_matrix)

    vector1 = U[:, 0] * np.sqrt(S[0])
    vector2 = U[:, 1] * np.sqrt(S[1])

    plt.plot([mean[0], mean[0] + vector1[0]], [mean[1], mean[1] + vector1[1]], color='red')
    plt.plot([mean[0], mean[0] + vector2[0]], [mean[1], mean[1] + vector2[1]], color='green')
    plt.show()


a()
# b()
# c()
# d()
# e()
# f()
