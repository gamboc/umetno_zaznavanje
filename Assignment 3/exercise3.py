import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

import a3_utils


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


def findedges(image, sigma, theta):
    mag, _ = gradient_magnitude(image, sigma)
    edges = np.where(mag >= theta, 1, 0)
    return edges


def non_max_suppression(image, sigma, theta):
    mag, dir = gradient_magnitude(image, sigma)
    edges = findedges(image, sigma, theta)
    mag = np.vstack((np.zeros((1, mag.shape[1])), mag, np.zeros((1, mag.shape[1]))))
    mag = np.hstack((np.zeros((mag.shape[0], 1)), mag, np.zeros((mag.shape[0], 1))))

    dir = np.where(dir < 0, math.pi+dir, dir)
    result = np.zeros(image.shape)
    for y, line in enumerate(image):
        for x, point in enumerate(line):
            if edges[y, x]:
                if dir[y, x] < math.pi/8 or dir[y, x] > 7*math.pi/8:
                    if mag[y+1, x+1] >= mag[y+1, x] and mag[y+1, x+1] >= mag[y+1, x+2]:
                        result[y, x] = 1
                elif dir[y, x] < 3*math.pi/8:
                    if mag[y+1, x+1] >= mag[y, x] and mag[y+1, x+1] >= mag[y+2, x+2]:
                        result[y, x] = 1
                elif dir[y, x] < 5*math.pi/8:
                    if mag[y+1, x+1] >= mag[y, x+1] and mag[y+1, x+1] >= mag[y+2, x+1]:
                        result[y, x] = 1
                else:
                    if mag[y+1, x+1] >= mag[y, x+2] and mag[y+1, x+1] >= mag[y+2, x]:
                        result[y, x] = 1

    return result


def hough_find_lines(image, theta_bins, rho_bins, threshold):
    A = np.zeros((rho_bins, theta_bins))

    diagonal = (image.shape[0]**2+image.shape[1]**2)**0.5
    theta = np.linspace(math.pi*(-1)/2, math.pi/2, num = theta_bins)
    for y, line in enumerate(image):
        for x, point in enumerate(line):
            if point != 0:
                rho = x*np.cos(theta) + y*np.sin(theta)
                rho = ((((rho+diagonal)/2)/diagonal)*rho_bins).astype(np.int32)
                A[rho, range(theta_bins)] += 1

    return A


def nonmaxima_suppression_box(feature):
    feature = np.vstack((np.zeros((1, feature.shape[1])), feature, np.zeros((1, feature.shape[1]))))
    feature = np.hstack((np.zeros((feature.shape[0], 1)), feature, np.zeros((feature.shape[0], 1))))

    for i, _ in enumerate(feature[1:len(feature)-1]):
        for j, _ in enumerate(feature[i, 1:len(feature[i])-1]):
            neighbourhood = feature[i:i+3, j:j+3]
            nbh_max = np.max(neighbourhood)
            feature[i:i+3, j:j+3] = np.where(neighbourhood == nbh_max, nbh_max, 0)
            got_one = False

            for k, _ in enumerate(feature[i:i+3]):
                for l, _ in enumerate(feature[k, j:j+3]):
                    if not got_one and feature[i+k, j+l] == nbh_max:
                        got_one = True
                    else:
                        feature[i+k, j+l] = 0

    return feature[1:len(feature)-1, 1:len(feature[1])-1]


def thresholdfeatures(features, threshold):
    features = np.where(features > threshold, 1, 0)
    thetas = []
    rhos = []
    for i, _ in enumerate(features):
        for j, _ in enumerate(features[i]):
            if features[i, j]:
                thetas.append(j)
                rhos.append(i)
    return thetas, rhos


def a():
    points = [(10, 10), (30, 60), (50, 20), (80, 90)]
    theta_bins, rho_bins = 300, 300

    thetas = np.linspace(-np.pi / 2, np.pi, num=theta_bins)
    cos = np.cos(thetas)
    sin = np.sin(thetas)

    for i in range(4):
        H = np.zeros((rho_bins, theta_bins))
        x = points[i][0]
        y = points[i][1]

        for theta in range(len(thetas)):
            rho = round(x * cos[theta] + y * sin[theta]) + rho_bins // 2
            H[rho, theta] += 1

        plt.subplot(2, 2, i + 1)
        plt.title("x = {}, y = {}".format(points[i][0], points[i][1]))
        plt.imshow(H)

    plt.show()


def b():
    synthetic = np.zeros((100, 100))
    synthetic[10][10] = 1
    synthetic[10][20] = 1

    image = hough_find_lines(synthetic, 300, 300, 0)

    plt.subplot(1, 3, 1)
    plt.title("synthetic")
    plt.imshow(image)

    plt.subplot(1, 3, 2)
    oneline = cv2.cvtColor(cv2.imread("./images/oneline.png"), cv2.COLOR_BGR2GRAY) / 255
    oneline = findedges(oneline, 0.5, 0.2)
    plt.title("oneline")
    plt.imshow(hough_find_lines(oneline, 300, 300, 0))

    plt.subplot(1, 3, 3)
    rectangle = cv2.cvtColor(cv2.imread("./images/rectangle.png"), cv2.COLOR_BGR2GRAY) / 255
    rectangle = findedges(rectangle, 0.5, 0.2)
    plt.title("rectangle")
    plt.imshow(hough_find_lines(rectangle, 300, 300, 0))

    plt.show()


def c():
    sigma = 0.5
    theta = 0.10

    synthetic = np.zeros((100, 100))
    synthetic[(10, 10), (10, 20)] = 1

    plt.subplot(1, 3, 1)
    synthetic = non_max_suppression(synthetic, sigma, theta)
    synthetic = nonmaxima_suppression_box(hough_find_lines(synthetic, 300, 300, theta))
    plt.imshow(synthetic)

    plt.subplot(1, 3, 2)
    oneline = cv2.cvtColor(cv2.imread("./images/oneline.png"), cv2.COLOR_BGR2GRAY) / 255
    oneline = non_max_suppression(oneline, sigma, theta)
    oneline = nonmaxima_suppression_box(hough_find_lines(oneline, 300, 300, theta))
    plt.imshow(oneline)

    plt.subplot(1, 3, 3)
    rectangle = cv2.cvtColor(cv2.imread("./images/rectangle.png"), cv2.COLOR_BGR2GRAY) / 255
    rectangle = non_max_suppression(rectangle, sigma, theta)
    rectangle = nonmaxima_suppression_box(hough_find_lines(rectangle, 300, 300, theta))
    plt.imshow(rectangle)

    plt.show()


def d():
    images = ['./images/oneline.png', './images/rectangle.png']
    sigma = 0.5
    threshold = 0.10
    rho_bins = 600
    theta_bins = 600

    synthetic = np.zeros((100, 100))
    synthetic[(10, 10), (10, 20)] = 1
    plt.subplot(1, 3, 1)
    plt.imshow(synthetic, cmap="gray")
    synthetic = non_max_suppression(synthetic, 1, threshold)
    featurespace = nonmaxima_suppression_box(hough_find_lines(synthetic, theta_bins, rho_bins, threshold))
    features = thresholdfeatures(featurespace, 3)

    for i, _ in enumerate(features[0]):
        theta = (features[0][i] / theta_bins - 0.5) * math.pi
        diagonal = (synthetic.shape[0] ** 2 + synthetic.shape[1] ** 2) ** 0.5
        rho = features[1][i] / rho_bins * diagonal * 2 - diagonal
        a3_utils.draw_line(rho, theta, synthetic.shape[0], synthetic.shape[1])

    for i, imageName in enumerate(images):
        image = cv2.imread(imageName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image.astype(np.float64) / 255
        plt.subplot(1, 3, i + 2)
        plt.imshow(image, cmap="gray")
        mag, _ = gradient_magnitude(synthetic, 0.5)
        image = non_max_suppression(mag, sigma, threshold)
        suppressed = nonmaxima_suppression_box(hough_find_lines(image, theta_bins, rho_bins, threshold))
        features = thresholdfeatures(suppressed, 100)

        for i, _ in enumerate(features[0]):
            theta = (features[0][i] / theta_bins - 0.5) * math.pi
            diagonal = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5
            rho = features[1][i] / rho_bins * diagonal * 2 - diagonal
            a3_utils.draw_line(rho, theta, image.shape[0], image.shape[1])

    plt.show()


def e():
    images = ['./images/bricks.jpg', './images/pier.jpg']
    sigma = 1
    threshold = 0.12
    rho_bins = 400
    theta_bins = 400

    for i, imageName in enumerate(images):
        image = cv2.imread(imageName)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float64) / 255
        suppressed = non_max_suppression(gray, sigma, threshold)
        featurespace = hough_find_lines(suppressed, theta_bins, rho_bins, threshold)

        plt.subplot(2, 2, i + 1)
        plt.title(imageName)
        plt.imshow(featurespace)

        plt.subplot(2, 2, i + 3)
        plt.imshow(image)

        featurespace = nonmaxima_suppression_box(featurespace)
        featurespace2 = np.reshape(featurespace, -1)
        featurespace2 = np.sort(featurespace2)
        print(featurespace2)
        thr = featurespace2[len(featurespace2) - 11]
        features = thresholdfeatures(featurespace, thr)

        for j, _ in enumerate(features[0]):
            theta = ((features[0][j] / theta_bins) * math.pi) - (math.pi / 2)
            diagonal = (image.shape[0] ** 2 + image.shape[1] ** 2) ** 0.5
            rho = features[1][j] / rho_bins * diagonal * 2 - diagonal
            a3_utils.draw_line(rho, theta, image.shape[0], image.shape[1])
    plt.show()

# a()
# b()
# c()
# d()
e()