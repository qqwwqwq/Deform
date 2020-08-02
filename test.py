# coding: utf-8
import json
from pandas import Series, DataFrame
import math
import random
import operator
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


def MCL(matrix,e,r):

    global qb
    global targ
    l=len(matrix)
    for i in range(l):
        matrix[i][i]=1
    print(matrix)
    yi=matrix/matrix.sum(axis=0)
    yi2=yi
    past=np.zeros((len(matrix), len(matrix)))
    sum=0
    while((past==yi2).all()==False and sum<10000):
        past=yi2
        yi2 =matrix_power(yi2,e)
        yi2 = np.power(yi2, r)
        yi2 = yi2 / yi2.sum(axis=0)
        sum=sum+1
        print(sum)

    clustr=[]
    print(clustr)
    for i in range(len(yi2)):
        yi2[i][i]=0
    for i in range(len(yi2)):
        a=[]
        for j in range(len(yi2)):
             if(yi2[j][i]>0):
                a.append(j)
        clustr.append(a)
    targ = [0] * len(clustr)
    final=[]
    for i in range(len(clustr)):
        if(targ[i]==0 and clustr[i]!=None):
            qb=[]
            findcluster(clustr,i,qb)
            final.append(qb)
    return final
def findcluster(clustr,i,qb):
    global targ
    if(targ[i]==0):
        qb.append(i)
        targ[i] = 1
        if (clustr[i] != None):
            for j in clustr[i]:
                findcluster(clustr, j, qb)
                for u in range(len(clustr)):
                    if j in clustr[u]:
                        findcluster(clustr,u,qb)
def ore(a):
    return math.atan((a[1])/(a[0]))/np.pi*360
def ang2(a,b):
    return abs(a.angle-b.angle)%180
def ang(a,b):
    if a[0]-b[0]==0:
        return 0
    else:
        return math.atan((a[1]-b[1])/(a[0]-b[0]))/np.pi*180

def bgr_rgb(img):
        (r, g, b) = cv2.split(img)
        return cv2.merge([b, g, r])
def distan(a,b):
    x=pow(abs(a[0]-b[0]),2)
    y=pow(abs(a[1]-b[1]),2)
    return sqrt(x+y)
def neighbor(p1,q1,p2,q2):
    global grapy_p
    global grapy_q
    s1=1
    s2=1
    d1=distan(p1.pt,p2.pt)
    d2=distan(q1.pt,q2.pt)
    for i in range(len(grapy_q)):
        if(grapy_q[p1.class_id][i]<d1):
            s1=s1+1
    for i in range(len(grapy_p)):
        if(grapy_p[q1.class_id][i]<d2):
            s2=s2+1

    # if(s1<30 and s2<6):
    #     return True
    # else:

    return s1,s2


def scale(p1,q1,p2,q2):
    l1=distan(p1.pt,p2.pt)
    l2=distan(q1.pt,q2.pt)
    threshold=1.5
    if(l1!=0 and l2 !=0):
        if (1 / threshold <= ((p1.size / l1) / (q1.size / l2)) <= threshold and 1 / threshold <= (
                (p2.size / l1) / (q2.size / l2)) <= threshold):

        # first=abs((p1.size / l1)/(q1.size / l2))
        #
        # second=abs((p2.size / l1)/(q2.size / l2))
            return True
        else:
            return False
##########################????????????????????????
def orientation(p1,q1,p2,q2):
    print(p1.pt,q1.pt,p2.pt,q2.pt)
    a=abs(ang(p1.pt,p2.pt)%180-ore(p1.pt)%180)
    b=abs(ang(q1.pt,q2.pt)%180-ore(q1.pt)%180)

    c=abs(ang(p1.pt,p2.pt)%180-ore(p2.pt)%180)
    d=abs(ang(q1.pt,q2.pt)%180-ore(q2.pt)%180)
    # a = abs(ang(p1.pt, p2.pt) % 180 - p1.angle)
    # b = abs(ang(q1.pt, q2.pt) % 180 - q1.angle)
    #
    # c = abs(ang(p1.pt, p2.pt) % 180 - p2.angle)
    # d = abs(ang(q1.pt, q2.pt) % 180 - q2.angle)
   # print("1111111111111111111", a, b, c, d)

    #if(abs(a-b)<=20 or abs(c-d)<=20):
    # if abs(a - b) <= 25 and abs(c-d) <=25:
    #     return True
    # else:
    #     return False
    return abs(a-b),abs(c-d)

def sift_detect(img1, img2):
    global grapy_p, sum
    global grapy_q
    kt1=[]
    kk1=[]
    kt2=[]
    kk2=[]
    kk3=[]
    kt5=[]
    kt6=[]
    kk4=[]
    symb=[]

    sift = cv2.xfeatures2d.SIFT_create()
    ###############################

    dst1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)

    dst2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)

    ##################################
    Maximg1 = np.max(img1)
    Minimg1 = np.min(img1)
    # 输出最小灰度级和最import operator大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a1 = float(Omax - Omin) / (Maximg1 - Minimg1)
    b1 = Omin - a1 * Minimg1
    # 线性变换
    O1 = a1 * img1 + b1
    O1 = O1.astype(np.uint8)
    Maximg2 = np.max(img2)
    Minimg2 = np.min(img2)
    # 输出最小灰度级和最大灰度级
    Omin, Omax = 0, 255
    # 求 a, b
    a2 = float(Omax - Omin) / (Maximg2 - Minimg2)
    b2 = Omin - a2 * Minimg2
    # 线性变换
    O2 = a2 * img2 + b2
    O2 = O2.astype(np.uint8)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute( img1 , None)
    kp2, des2 = sift.detectAndCompute( img2 , None)
    FLANN_INDEX_KDTREE = 0
    # BFMatcher with default params
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2,k=2)

    # Apply ratio test


    good = [[m] for m, n in matches if m.distance < 0.8* n.distance]
    #build grapy
    grapy_all=np.zeros((len(good), len(good)))
    grapy_p = np.zeros((len(good), len(good)))
    grapy_q = np.zeros((len(good), len(good)))
    for i in range(len(good)):
        for j in range(len(good)):
            p1=kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            kp1[good[i][0].queryIdx].class_id = i
            kp2[good[i][0].trainIdx].class_id = i
            p2=kp1[good[j][0].queryIdx]
            q2=kp2[good[j][0].trainIdx]
            grapy_p[i][j]=distan(p1.pt,p2.pt)
            grapy_q[i][j]=distan(q1.pt,q2.pt)
            grapy_p[j][i] = distan(p1.pt, p2.pt)
            grapy_q[j][i] = distan(q1.pt, q2.pt)
    for i in range(len(good)):
        for j in range(len(good)):
            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            # if(neighbor(p1,q1,p2,q2)):
            #         if(orientation(p1,q1,p2,q2)):
            #     if(scale(p1,q1,p2,q2)):
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            kt1.append(n1)
            kt2.append(n2)
            kt5.append(o1)
            kt6.append(o2)
    m1=max(kt1)
    m11=min(kt1)
    m2=max(kt2)
    m22=min(kt2)
    m5 = max(kt5)
    m55 = min(kt5)
    m6 = max(kt6)
    m66 = min(kt6)
    throsld = len(kt1)
    for i in range(len(good)):
        for j in range(len(good)):
            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            n11=(n1-m11)*throsld/(m1-m11)
            n22 = (n2 - m22) * throsld / (m2 - m22)
            o11=(o1-m55)*throsld/(m5-m55)
            o22 = (o2 - m66) * throsld/ (m6 - m66)
            t = (n11 + n22 + o11 + o22)
            symb.append(t)
    maxs=max(symb)
    mins=min(symb)

    for i in range(len(good)):
        for j in range(i+1,len(good)):
            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            n11=(n1-m11)*throsld/(m1-m11)
            n22 = (n2 - m22) * throsld / (m2 - m22)
            o11=(o1-m55)*throsld/(m5-m55)
            o22 = (o2 - m66) * throsld/ (m6 - m66)
            t =(n11+n22+o11+o22-mins)*throsld/(maxs-mins)
            kk1.append(n11+n22)
            # kk2.append(s11+s22)
            kk3.append(o11+o22)
            kk4.append(t)
            #print("this s neighbor",neighbor(p1, q1, p2, q2), scale(p1, q1, p2, q2), orientation(p1, q1, p2, q2), t)

##############################################################################8####################################
            # if n1+n2<20 and s1+s2<4 and o1+o2<60:
            step=throsld/8
            if scale(p1, q1, p2, q2):
                if t > 0 and t < step:
                    print(throsld)
                    grapy_all[i][j] = 1/t
                    grapy_all[j][i] = 1/t
                elif t > step:
                    grapy_all[i][j] = 1 / (int(t / step) * step)
                    grapy_all[j][i] = 1 / (int(t / step) * step)

    print(grapy_q)
    print(grapy_p)
    a=[]
    for i in good:
        sum=0
        for j in range(len(grapy_all)):
            c=kp1[i[0].queryIdx].class_id
            if(grapy_all[c][j]==1):
                sum=sum+1
                if(sum>=1):
                    a.append(i)
    ######
    final=[]
    bob=MCL(grapy_all,2,2.5)#get cluster.....................................
    lens=[]
    long=2
    for i in bob:
        if len(i)>=2:
            long=3
    if(len(bob)>1):
        for j in bob:
            lens.append(len(j))
        for i in range(7):
            for q in range(len(lens)):
                if lens[q] == max(lens):
                    if(len(bob[q])>=long):
                        final.append(bob[q])
                        lens[q] = 0
                    break
    else:
        for i in bob:
            if len(i)>2:
                final.append(i)

    apple=[]
    for u in range(len(final)):
        pp=[]
        for i in good:
            c = kp1[i[0].queryIdx].class_id
            print(c)
            if(c in final[u]):
                pp.append(i)
        apple.append(pp)
    print("final",apple)
    # cv2.drawMatchesKnn expects list of lists as matches.
    if len(apple)<1:
        print("not good")
 ################################find circul
    imgg =drawMatches(img1, kp1, img2, kp2, apple)
    img2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, (0, 0, 255), (255, 0, 0), None, flags=2)
    # print(k1)
    ww=[1,2,3,4,5,6,7,8,9,10,100]
    s = Series(kk1)
    ss2=Series(kk2)
    ss3 = Series(kk3)
    ss4 = Series(kk4)
    plt.subplot(221)
    plt.hist(s,  bins=400, color='r', density=True, range=(-1, throsld))
    plt.subplot(222)
    plt.hist(ss2,  bins=400, color='r',density=True,range=(-1,throsld))
    plt.subplot(223)
    plt.hist(ss3,  bins=200, color='r', density=True, range=(-1, throsld))
    plt.subplot(224)
    plt.hist(ss4,  bins=400, color='r', density=True, range=(-1, throsld))
    plt.show()
    print(kk1[:10])
    print(kk2[:10])
    print(kk3[:10])
    #
    return bgr_rgb(imgg/255)


if __name__ == "__main__":
    # load image
    image_a = cv2.imread('/home/hexin/桌面/deform/news2.png')
    image_b = cv2.imread('/home/hexin/桌面/cool2.png')
    image_d = cv2.imread('/home/hexin/桌面/depth/cool2.png')
    #################################################nn
##########d
    print(image_b.size)
    img2= sift_detect(image_a, image_b)
    print("oooooooooooooooooooook")
  #  plt.subplot(121)
    plt.imshow(img2)
    plt.show()
