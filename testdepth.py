import random

import pyk4a
import cv2
import numpy as np


def vector(p1, p2, p3):
    a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
    b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
    c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    l = np.sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))
    return [a / l, b / l, c / l]
index = random.sample(range(0, 100), 3)
print(index)
print(vector([-67, -35, 283], [-65, -33, 283] ,[-65, -30, 282]))
