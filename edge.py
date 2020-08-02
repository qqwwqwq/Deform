import math
import random
from math import sqrt
from numpy.linalg import matrix_power

from draw import drawMatches
import matplotlib
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import cv2
ww=np.load('/home/hexin/桌面/deform/boxd.npy')

fil = np.array([[ -1,-1, -1],                        #这个是设置的滤波，也就是卷积核
                [ -1, 8, -1],
                [  -1, -1, -1]])

res = cv2.filter2D(ww,-1,fil)
res=res*100
#canny=cv2.Canny(ww,50,150)
print(res)

plt.imshow(res)
plt.show()