import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def myhist3(image, n_bins):
    histogram = np.zeros((n_bins, n_bins, n_bins))
    image = (image / 255).astype(np.float64)
    image = (image * n_bins).astype(np.uint8)

    for i in range(len(image)):
        for j in range(len(image[i])):
            histogram[int(np.floor(image[i][j][0]))][int(np.floor(image[i][j][1]))][int(np.floor(image[i][j][2]))] += 1

    histogram /= np.sum(histogram)

    return histogram


def compare_histograms(hist1, hist2, string):
    if string == "L2":
        return L2(hist1, hist2)
    elif string == "chi":
        return chi_square(hist1, hist2)
    elif string == "intersection":
        return intersection(hist1, hist2)
    else:
        return hellinger(hist1, hist2)


def L2(hist1, hist2):
    return np.sum((hist1 - hist2)**2)**0.5


def chi_square(hist1, hist2):
    return 0.5 * np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))


def intersection(hist1, hist2):
    return 1 - np.sum(np.minimum(hist1, hist2))


def hellinger(hist1, hist2):
    return np.sqrt(np.sum((np.sqrt(hist1) - np.sqrt(hist2)) ** 2) / 2)


def retrieve(path, n_bins):
    files = os.listdir(path)
    histograms = []
    for i in range(len(files)):
        histograms.append((files[i], myhist3(cv2.cvtColor(cv2.imread(path + "/" + files[i]), cv2.COLOR_BGR2RGB), n_bins)))

    return histograms


def c():
    image1 = cv2.cvtColor(cv2.imread("./dataset/object_01_1.png"), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread("./dataset/object_02_1.png"), cv2.COLOR_BGR2RGB)
    image3 = cv2.cvtColor(cv2.imread("./dataset/object_03_1.png"), cv2.COLOR_BGR2RGB)

    hist1 = myhist3(image1, 8)
    hist2 = myhist3(image2, 8)
    hist3 = myhist3(image3, 8)

    plt.subplot(2, 3, 1)
    plt.imshow(image1)
    plt.subplot(2, 3, 2)
    plt.imshow(image2)
    plt.subplot(2, 3, 3)
    plt.imshow(image3)

    reshaped1 = hist1.reshape(-1)
    reshaped2 = hist2.reshape(-1)
    reshaped3 = hist3.reshape(-1)

    indexes = range(1, len(reshaped1) + 1)

    plt.subplot(2, 3, 4)
    plt.title(np.around(L2(hist1, hist1), decimals=2))
    plt.plot(indexes, reshaped1)
    plt.subplot(2, 3, 5)
    plt.title(np.around(L2(hist1, hist2), decimals=2))
    plt.plot(indexes, reshaped2)
    plt.subplot(2, 3, 6)
    plt.title(np.around(L2(hist1, hist3), decimals=2))
    plt.plot(indexes, reshaped3)

    print("L2 distance: " + str(compare_histograms(hist1, hist1, "L2")) + " " + str(compare_histograms(hist1, hist2, "L2")) + " " + str(compare_histograms(hist1, hist3, "L2")))
    print("Chi² distance: " + str(compare_histograms(hist1, hist1, "chi")) + " " + str(compare_histograms(hist1, hist2, "chi")) + " " + str(compare_histograms(hist1, hist3, "chi")))
    print("Intersection distance: " + str(compare_histograms(hist1, hist1, "intersection")) + " " + str(compare_histograms(hist1, hist2, "intersection")) + " " + str(compare_histograms(hist1, hist3, "intersection")))
    print("Hellinger distance: " + str(compare_histograms(hist1, hist1, "hellinger")) + " " + str(compare_histograms(hist1, hist2, "hellinger")) + " " + str(compare_histograms(hist1, hist3, "hellinger")))

    plt.show()


def d():
    measures = ["L2", "chi", "intersection", "hellinger"]
    histograms = retrieve("./dataset", 5)

    for m in measures:
        distances = []
        for i in range(len(histograms)):
            distances.append((histograms[i][0], compare_histograms(histograms[0][1], histograms[i][1], m)))

        sorted_distances = distances.copy()
        sorted_distances.sort(key=lambda y: y[1])

        for i in range(6):
            name = sorted_distances[i][0]
            image = cv2.cvtColor(cv2.imread('./dataset/' + name), cv2.COLOR_BGR2RGB)

            histogram = [item for item in histograms if item[0] == name][0][1]
            unrolled = histogram.reshape(-1)
            indexes = range(1, len(unrolled) + 1)

            plt.subplot(2, 6, i + 1)
            plt.title(name)
            plt.imshow(image)

            plt.subplot(2, 6, i + 7)
            plt.title(sorted_distances[i][1])
            plt.bar(indexes, unrolled, width=5)
        plt.show()


def e():
    measures = ["L2", "chi", "intersection", "hellinger"]
    histograms = retrieve("./dataset", 5)

    for m in measures:
        distances = []
        for i in range(len(histograms)):
            distances.append((histograms[i][0], compare_histograms(histograms[0][1], histograms[i][1], m)))

        sorted_distances = distances.copy()
        sorted_distances.sort(key=lambda y: y[1])

        indexes = range(1, len(distances) + 1)
        closest_values = []
        closest_indexes = []
        for i in range(6):
            closest_values.append(sorted_distances[i][1])

        for i in range(6):
            for j in range(len(distances)):
                if closest_values[i] == distances[j][1]:
                    closest_indexes.append(j)

        array, array_sorted = [], []
        for i in range(len(distances)):
            array.append(distances[i][1])
            array_sorted.append(sorted_distances[i][1])

        array, array_sorted = np.array(array), np.array(array_sorted)

        plt.subplot(1, 2, 1)
        plt.plot(indexes, array)
        plt.plot(closest_indexes, closest_values, 'ro', mfc='none')

        plt.subplot(1, 2, 2)
        plt.plot(indexes, array_sorted)
        plt.plot(range(6), closest_values, 'ro', mfc='none')
        plt.show()

# c()
# d()
# e()
