import numpy as np
import cv2
from matplotlib import pyplot as plt
import a5_utils


def b():
    f = 0.0025
    T = 0.12

    pz_range = np.linspace(0.1, 10, num=100)
    d = (f / pz_range) * T

    plt.plot(d)
    plt.show()

b()