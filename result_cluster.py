
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
class clater():
    def __init__(self,dep1,direct1,locate1,range1,d1,maxx1,minx1,twod1,maxy1,miny1):
        self.dep=dep1
        self.direct=direct1
        self.locate=locate1
        self.range=range1
        self.d=d1
        self.maxx=maxx1
        self.minx=minx1
        self.teod=twod1
        self.maxy=maxy1
        self.miny=miny1


# ww=np.load('/home/hexin/桌面/deform/d1.npy')
#
# print(ww)
# plt.imshow(ww)
# plt.show()