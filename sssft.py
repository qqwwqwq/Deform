# coding: utf-8
import json
import uuid

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.abc import x,y,z
from result_cluster import clater
from T2to3 import T2to3
import math
import random
from math import sqrt
from numpy.linalg import matrix_power
import tkinter
from draw import drawMatches
import matplotlib
import pyk4a
from pyk4a import Config, PyK4A, ColorResolution
import numpy as np
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')
import cv2
global rotatevalue1
global rotatevalue2



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
        return math.atan((a[1]-b[1])/(a[0]-b[0]))/np.pi*360

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
    threshold=1.4
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


def sift_detect(img1, img2,imgd,imgdo):
    global grapy_p
    global grapy_q
    kt1 = []
    kk1 = []
    kt2 = []
    kk2 = []
    kk3 = []
    kt5 = []
    kt6 = []
    kk4 = []
    symb = []
    sift = cv2.xfeatures2d.SIFT_create()
    ###############################

    dst1 = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)

    dst2 = cv2.fastNlMeansDenoisingColored(img2, None, 10, 10, 7, 21)

    ##################################
    Maximg1 = np.max(img1)
    Minimg1 = np.min(img1)
    # 输出最小灰度级和最大灰度级
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


    good = [[m] for m, n in matches if m.distance < 0.7* n.distance]

    #build grapy
    grapy_all=np.zeros((len(good), len(good)))
    grapy_p = np.zeros((len(good), len(good)))
    grapy_q = np.zeros((len(good), len(good)))
    for i in range(len(good)):
        for j in range(i+1,len(good)):
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
        for j in range(i+1,len(good)):
            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            kt1.append(n1)
            kt2.append(n2)
            kt5.append(o1)
            kt6.append(o2)
    m1 = max(kt1)
    m11 = min(kt1)
    m2 = max(kt2)
    m22 = min(kt2)
    m5 = max(kt5)
    m55 = min(kt5)
    m6 = max(kt6)
    m66 = min(kt6)
    throsld = len(kt1)
    '''
    kp1: keypointset in first image
    kp2:keypointset in second image
    element in good: the match of a pair of the key point 
    queryIdx: index in kp1
    trianIdx: index in kp2
    '''
    for i in range(len(good)):
        for j in range(i+1, len(good)):

            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            n11 = (n1 - m11) * throsld / (m1 - m11)
            n22 = (n2 - m22) * throsld / (m2 - m22)
            o11 = (o1 - m55) * throsld / (m5 - m55)
            o22 = (o2 - m66) * throsld / (m6 - m66)
            t = ((n11 + n22) + o11 + o22)
            symb.append(t)
    maxs = max(symb)
    mins = min(symb)

    for i in range(len(good)):
        for j in range(i+1,len(good)):
            p1 = kp1[good[i][0].queryIdx]
            q1 = kp2[good[i][0].trainIdx]
            p2 = kp1[good[j][0].queryIdx]
            q2 = kp2[good[j][0].trainIdx]
            n1, n2 = neighbor(p1, q1, p2, q2)
            o1, o2 = orientation(p1, q1, p2, q2)
            n11 = (n1 - m11) * throsld / (m1 - m11)
            n22 = (n2 - m22) * throsld / (m2 - m22)
            o11 = (o1 - m55) * throsld / (m5 - m55)
            o22 = (o2 - m66) * throsld / (m6 - m66)
            t = ((n11 + n22) + o11 + o22 - mins) * throsld / (maxs - mins)
            kk1.append(n11 + n22)
            # kk2.append(s11+s22)
            kk3.append(o11 + o22)
            kk4.append(t)
            #print("this s neighbor", neighbor(p1, q1, p2, q2), scale(p1, q1, p2, q2), orientation(p1, q1, p2, q2), t)

            ##############################################################################8####################################
            # if n1+n2<20 and s1+s2<4 and o1+o2<60:
            step = throsld / 8
            if scale(p1, q1, p2, q2):
                if t > 0 and t < step:
                    #print(throsld)
                    grapy_all[i][j] = 1
                    grapy_all[j][i] = 1
                elif t > step:
                    grapy_all[i][j] = 1 / (int(t / step) * step)
                    grapy_all[j][i] = 1 / (int(t / step) * step)
    print(grapy_q)
    print(grapy_p)
    # a=[]
    # for i in good:
    #     sum=0
    #     for j in range(len(grapy_all)):
    #         c=kp1[i[0].queryIdx].class_id
    #         if(grapy_all[c][j]==1):
    #             sum=sum+1
    #             if(sum>=1):
    #                 a.append(i)
    ######
    final=[]
    bob = MCL(grapy_all, 2, 2) ##get cluster
    lens=[]
    long=2
    for i in bob:
        if len(i)>=2:
            long=3
    if(len(bob)>1):
        for j in bob:
            lens.append(len(j))
        for i in range(6):
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
    # cv2.drawMatchesKnn expects list of list   s as matches.
    if len(apple)<1:
        print("not good")
    ori_out=[]
    process_out=[]
    for n in apple:

        out_d,out_o=findring(n,kp1,kp2,imgdo,imgd)
        if out_d!= 0 and out_o!=0:
            ori_out.append(out_o)
            process_out.append(out_d)

        #################################find circul
    #bigest=rangeofring2.index((max(rangeofring2)))

    imgg =drawMatches(img1, kp1, img2, kp2, apple)
    img2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, (0, 0, 255), (255, 0, 0), None, flags=2)
    return bgr_rgb(imgg/255),process_out,ori_out

def findring(area,kp1,kp2,imgd1,imgd2):

    def three(p1, p2, p3):
        k = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - k * p1[0]
        return abs((k * p3[0] - p3[1] + b) / sqrt(pow(k, 2) + 1))
    def mdcircle(p1, p2):
        center = []
        center.append((p1[0] + p2[0]) / 2)
        center.append((p1[1] + p2[1]) / 2)
        #R = sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2)) / 2
        R=30
        return center, R

    def vector1(p1, p2, p3):
        a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
        b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
        c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        l = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))
        if c < 0:
            c = c * (-1)
            a = a * (-1)
            b = b * (-1)
        if l != 0:
            return [round(a / l,4), round(b / l,4), round(c / l,4)]
        else:
            return [0, 0, 0]
    def vector(p1,p2,p3):
        a= ((p2[1]-p1[1])*(p3[2]-p1[2])-(p2[2]-p1[2])*(p3[1]-p1[1]))
        b=( (p2[2]-p1[2])*(p3[0]-p1[0])-(p2[0]-p1[0])*(p3[2]-p1[2]))
        c=( (p2[0]-p1[0])*(p3[1]-p1[1])-(p2[1]-p1[1])*(p3[0]-p1[0]))
        l=sqrt(pow(a,2)+pow(b,2)+pow(c,2))
        if c<0:
            c=c*(-1)
            a=a*(-1)
            b=b*(-1)
        q=np.array([a,b,c])
        if l!=0:
            return np.around(q/l, decimals=3)
        else:
            return [0,0,0]
    def circle(p1, p2, p3):
        x1 = p1[0]
        x2 = p2[0]
        x3 = p3[0]
        y1 = p1[1]
        y2 = p2[1]
        y3 = p3[1]
        a = x1 - x2
        b = y1 - y2
        c = x1 - x3
        d = y1 - y3
        a1 = ((x1 * x1 - x2 * x2) + (y1 * y1 - y2 * y2)) / 2.0
        a2 = ((x1 * x1 - x3 * x3) + (y1 * y1 - y3 * y3)) / 2.0
        theta = b * c - a * d
        if abs(theta) < 1e-7:
            return -1
        x0 = (b * a2 - d * a1) / theta
        y0 = (c * a1 - a * a2) / theta
        r = np.sqrt(pow((x1 - x0), 2) + pow((y1 - y0), 2))
        e = []
        e.append(x0)
        e.append(y0)
        return e, r


    # maxlen = 0
    # area = []
    # for i in apple:
    #     if len(i)>maxlen:
    #         maxlen=len(i)
    #         area=i
    #if in the circle range
    pointset1=[]
    pointset2=[]
    for q in area:
        pointset2.append(kp2[q[0].trainIdx].pt)
        pointset1.append(kp1[q[0].queryIdx].pt)

    def processpointset(pointset,depimg):
        sign = [0] * len(pointset)
        for i in pointset:
            print(depimg[int(i[1])][int(i[0])])
        print("++++++++++++", pointset)
        ###distance matirx
        grapy_distance = np.zeros((len(pointset), len(pointset)))
        for i in range(len(grapy_distance)):
            for j in range(len(grapy_distance)):
                grapy_distance[i][j] = distan(pointset[i], pointset[j])
        # print(grapy_distance)
        # find longest distance
        longest = 0
        poin1 = poin2 = -1
        for i in range(len(grapy_distance)):
            for j in range(len(grapy_distance)):
                if grapy_distance[i][j] > longest:
                    longest = grapy_distance[i][j]
                    poin1 = i
                    poin2 = j
        sign[poin1] = 1
        sign[poin2] = 2
        print(poin1, poin2)
        centerr, r = mdcircle(pointset[poin1], pointset[poin2])
        # print("ringg", centerr, r)
        # for i in range(len(pointset)):
        #     if distan(pointset[i], centerr) <= r:
        #         sign[i] = 1
        # if min(sign) == 0:
        #     one_max = 0
        #     taget = -1
        #     for j in range(len(sign)):
        #         if sign[j] == 0:
        #             if three(pointset[poin1], pointset[poin2], pointset[j]) > one_max:
        #                 one_max = three(pointset[poin1], pointset[poin2], pointset[j])
        #                 taget = j
        #                 print("007")
        #     # print(pointset[poin1],pointset[poin2],pointset[taget])
        #     centerr, r = circle(pointset[poin1], pointset[poin2], pointset[taget])
        #     print(centerr, r)
        finalset = []  # depth image pixel
        cx = 2044.08
        cy = 1550.39
        fx = 1955.83
        fy = 1955.42
        xaxis = int(centerr[1])##
        yaxis = int(centerr[0])##inverse x,y pixel
        zw1 = int(depimg[xaxis][yaxis])
        print(zw1,"zwwwwwwwwwwwwww")
        while zw1 == 0:
            xaxis += 1
            yaxis += 1
            zw1 = depimg[xaxis][yaxis]
        xw1 = int((xaxis - cx) * zw1 / fx)
        yw1 = int((yaxis - cy) * zw1 / fy)
        ccord = [xw1,yw1,zw1]
        print(ccord, 'qqqqqqqqqqqqqqqqqqqqmmmmmmmmmmmmmmmm')
        black = []

        for i in range(int(centerr[0] - r), int(centerr[0] + r)):
            for j in range(int(centerr[1] - r), int(centerr[1] + r)):
                if depimg[j][i] != 0:
                    zw = int(depimg[j][i])
                    xw = int((j - cx) * zw / fx)
                    yw = int((i - cy) * zw / fy)
                    yes = [xw, yw, zw]
                    finalset.append(yes)
            # j1=int(sqrt(pow(r,2)-pow(centerr[0]-i,2))+centerr[1])
            # j2=int(centerr[1]-sqrt(pow(r,2)-pow(centerr[0]-i,2)))
            # if depimg[j1][i] != 0:
            #     zw = int(depimg[j1][i])
            #     xw = int((j1 - cx) * zw / fx)
            #     yw = int((i - cy) * zw / fy)
            #     yes = [xw, yw, zw]
            #     finalset.append(yes)
            # if depimg[j2][i] != 0:
            #     zw2 = int(depimg[j2][i])
            #     xw2 = int((j2 - cx) * zw2 / fx)
            #     yw2 = int((i - cy) * zw2 / fy)
            #     yes2 = [xw2, yw2, zw2]
            #     finalset.append(yes2)


                    # print("yes",finalset)
        if finalset==[]:
            return 0
        # chiek = []
        # for i in finalset:
        #     print(2223)
        #     if i not in chiek:
        #         print(222)
        #         chiek.append(i)
        # print("------------", chiek)

        vecooo = []
        for i in range(200):
            index = random.sample(range(0, len(finalset)),3)

            directn=vector1(finalset[index[0]], finalset[index[1]],finalset[index[2]])
            if directn!=[0,0,0]:
                vecooo.append( directn)
                print( directn)
        print("vvvvvvvvvvvvvv", vecooo)
        sup = np.array(vecooo)
        sock = sup.mean(axis=0)
        vecooo2=[]
        for i in vecooo:
            #if angelsurface(i,sock)<10:
            vecooo2.append(i)
        sup2=np.array(vecooo2)
        sock1=sup2.mean(axis=0)
        print(sock1)
        sock2 = []
        for i in sock1:
            #i=i / sqrt(pow(sock[0],2)+pow(sock[1],2)+pow(sock[2],2))
            j=np.around(i,decimals=3)
            sock2.append(j)
        print(sock2)
        cv2.circle(depimg,(int(centerr[0]),int(centerr[1])),int(r),(0,0,255),thickness=3)
        # for i in black:
        #     depimg[i[1]][i[0]] = 0
        out=clater(depimg,sock2,ccord,r)
        return out

    out_d=processpointset(pointset2,imgd2)
    print("second")
    out_o= processpointset(pointset1, imgd1)
    print("first")
    return out_d,out_o

def ini():
    btn1.pack_forget()
    btnx.pack()
    btnx2.pack()
    btnx3.pack()
    btnx4.pack()
    btnx5.pack()
    btnx6.pack()

    ax.set_xlim3d(-300, 300)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.set_ylim3d(-300, 300)
    ax.set_zlim3d(0, 600)

    ax.cla()
    zc=0
    for i in range(len(ori_out)):
        X = []
        Y = []
        Z = []
        X.append(ori_out[i].locate[0])
        Y.append(ori_out[i].locate[1])
        Z.append(ori_out[i].locate[2])
        X.append(ori_out[i].locate[0]+4*ori_out[i].direct[0])
        Y.append(ori_out[i].locate[1] + 4*ori_out[i].direct[1])
        Z.append(ori_out[i].locate[2] + 4*ori_out[i].direct[2])
        ax.plot(X, Y, Z, 'bo--', color="black",linestyle=":")
    for i in range(len(ori_out)):
        for j in range(i,len(ori_out)):
            X = []
            Y = []
            Z = []
            X.append(ori_out[i].locate[0])
            Y.append(ori_out[i].locate[1])
            Z.append(ori_out[i].locate[2])
            X.append(ori_out[j].locate[0])
            Y.append(ori_out[j].locate[1])
            Z.append(ori_out[j].locate[2])
            ax.plot(X, Y, Z, 'bo--', color=aric[zc])
            zc+=1

    canvs.draw()



def oriform():

    #ax.set_xlim3d(-300, 300)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
   # ax.set_ylim3d(-300, 300)
   # ax.set_zlim3d(0, 400)

    zc = 0
    ax.cla()
    for i in range(len(ori_out)):
        X = []
        Y = []
        Z = []
        X.append(ori_out[i].locate[0])
        Y.append(ori_out[i].locate[1])
        Z.append(ori_out[i].locate[2])
        X.append(ori_out[i].locate[0]+4*ori_out[i].direct[0])
        Y.append(ori_out[i].locate[1] + 4*ori_out[i].direct[1])
        Z.append(ori_out[i].locate[2] + 4*ori_out[i].direct[2])
        ax.plot(X, Y, Z, 'bo--', color="black",linestyle=":")
    for i in range(len(ori_out)):
        for j in range(i, len(ori_out)):
            X = []
            Y = []
            Z = []
            X.append(ori_out[i].locate[0])
            Y.append(ori_out[i].locate[1])
            Z.append(ori_out[i].locate[2])
            X.append(ori_out[j].locate[0])
            Y.append(ori_out[j].locate[1])
            Z.append(ori_out[j].locate[2])
            ax.plot(X, Y, Z, 'bo--', color=aric[zc])
            zc += 1

    canvs.draw()
def randomcolor():
    colora=["1",'2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color=""
    for i in range(6):
        color+=colora[random.randint(0,14)]
    return "#"+color

def proform():

    ax.set_xlim3d(-300, 300)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_ylim3d(-300, 300)
    ax.set_zlim3d(0, 300)
    zc=0
    ax.cla()
    for i in range(len(process_out)):
        X = []
        Y = []
        Z = []
        X.append(process_out[i].locate[0])
        Y.append(process_out[i].locate[1])
        Z.append(process_out[i].locate[2])
        X.append(process_out[i].locate[0] + 4 * process_out[i].direct[0])
        Y.append(process_out[i].locate[1] + 4 * process_out[i].direct[1])
        Z.append(process_out[i].locate[2] + 4 * process_out[i].direct[2])
        ax.plot(X, Y, Z, 'bo--', color="black", linestyle=":")
    for i in range(len(process_out)):
        for j in range(i, len(process_out)):
            X = []
            Y = []
            Z = []
            X.append(process_out[i].locate[0])
            Y.append(process_out[i].locate[1])
            Z.append(process_out[i].locate[2])
            X.append(process_out[j].locate[0])
            Y.append(process_out[j].locate[1])
            Z.append(process_out[j].locate[2])
            ax.plot(X, Y, Z, 'bo--', color=aric[zc])
            zc += 1

    canvs.draw()

def rt1():
    global rotatevalue1
    global rotatevalue2
    rotatevalue1 +=5
    ax.view_init(rotatevalue1, rotatevalue2)
def rt2():
    global rotatevalue1
    global rotatevalue2
    rotatevalue1 -= 5
    ax.view_init(rotatevalue1, rotatevalue2)
def rt3():
    global rotatevalue1
    global rotatevalue2
    rotatevalue2 += 5
    ax.view_init(rotatevalue1, rotatevalue2)
def rt4():
    global rotatevalue1
    global rotatevalue2
    rotatevalue2 -= 5
    ax.view_init(rotatevalue1, rotatevalue2)

def angelsurface(a,b):
    cosang=(a[0]*b[0]+a[1]*b[1]+a[2]*b[2])/(sqrt(pow(a[0],2)+pow(a[1],2)+pow(a[2],2))*sqrt(pow(b[0],2)+pow(b[1],2)+pow(b[2],2)))

    angofsur=math.acos(round(cosang,3))
    return  angofsur*180/np.pi
def bending(c,d):
    a=c.direct
    b=d.direct
    bp = d.locate
    ap=c.locate
    da=np.dot(np.array(a),np.array(ap))
    db=np.dot(np.array(b),np.array(bp))
    print(da,db,"ddd")
    px=ap[0]
    aa = solve([px*a[0] + y*a[1] +z*a[2]- da, px * b[0] + b[1] * y +z*b[2]- db], [y, z])
    gap=[px,aa[y],aa[z]]
    x1=a[1]*b[2]-b[1]*a[2]
    y2=a[2]*b[0]-a[0]*b[2]
    z3=a[0]*b[1]-a[1]*b[0]
    t=[x1,y2,z3]
    return np.array(t),np.array(gap)
t,gap=bending(c,d)
def slide(t,gap,start,end,planea,planeb):
    alpa = 2
    gapa=planea.direct
    gapb=planeb.direct
    da=np.dot(np.array(planea.direct),np.array(planea.locate))
    db=np.dot(np.array(planeb.direct),np.array(planeb.locate))
    def cal(point):
        state=(point-gap[0])/t[0]
        y=state*t[1]+gap[1]
        z=state*t[2]+gap[2]
        return [point,y,z]
    for i in range(start,end-alpa):
        point1=cal(i)
        point2=cal(i+alpa)
        print(i,gapa,point1,t)
        cc=solve([gapa[0]*x+gapa[1]*y+gapa[2]*z-da,(x-point1[0])**2+(y-point1[1])**2+(z-point1[2])**2-4,t[0]*(point1[0]-x)+t[1]*(point1[1]-y)+t[2]*(point1[2]-z)],[x,y,z])
        print(i,cc)


if __name__ == "__main__":
    # load image
    image_a = cv2.imread('/home/hexin/桌面/deform/news3.png')
    image_b = cv2.imread('/home/hexin/桌面/dataset/tea2.png')
    image_d = np.load('/home/hexin/桌面/deform/newsd3.npy')
    ###########################################################
    k4a = PyK4A(Config(color_resolution=ColorResolution.RES_3072P,
                       depth_mode=pyk4a.DepthMode.WFOV_UNBINNED,
                       synchronized_images_only=True,
                       camera_fps=pyk4a.FPS.FPS_5, ))
    k4a.connect()
    k4a.exposure_mode_auto=True
    k4a.whitebalance_mode_auto=True
    k4a.sharpness=4
    k4a.backlight_compensation=1
    k4a.gain=100

    # getters and setters directly get and set on device

    #k4a.whitebalance = 4510
    #assert k4a.whitebalance == 4510
    #a = json.loads(k4a.get_calibra())
    #inner=a["CalibrationInformation"]['Cameras'][0]['Intrinsics']["ModelParameters"]

    while 1:
        # img_color = k4a.get_capture(color_only=True)
        img_color, img_depth = k4a.get_capture() # Would also fetch the depth image
        im = np.asarray(img_depth, np.uint16)
        #im2=np.asarray(image_d, np.uint16)
        cv2.imshow('k4a2', img_color)
        if cv2.waitKey(1) & 0xff == 27:
            cv2.destroyAllWindows()
            #cv2.imwrite('/home/hexin/桌面/cool2.png', img_color)
            print(len(image_a), len(img_color[:, :, :3]), len(img_depth),len(image_d))
            cv2.imwrite('/home/hexin/桌面/cool2.png', img_color[:, :, :3])
            #img2,xxx,s,center= sift_detect(image_a, img_color[:, :, :3], im)
            img2, process_out,ori_out = sift_detect(image_a, img_color[:, :, :3], im,image_d)
            for q in range(len(process_out)):
                print("origin",ori_out[q].direct,ori_out[q].locate)
                print("box",process_out[q].direct,  process_out[q].locate)
            grapy_angel = np.zeros((len(process_out), len(process_out)))
            grapy_bend = np.zeros((len(process_out), len(process_out)))
            for i in range(len(process_out)):
                for j in range(i+1,len(process_out)):
                    print(i,j,bending(process_out[i],process_out[j]))
                    grapy_angel[i][j]=angelsurface(process_out[i].direct,process_out[j].direct)
                    grapy_angel[j][i] = angelsurface(process_out[i].direct, process_out[j].direct)
            print(grapy_angel)
            xxx=process_out[-1].dep
            ooo=ori_out[-1].dep
            aric=[]
            for i in range(int((len(ori_out)-1)*len(ori_out))):
                aric.append(randomcolor())
            rotatevalue1=5
            rotatevalue2=5
            fig = plt.figure()
            #ax = Axes3D(fig)
            ax2 = fig.add_subplot(221)
            ax2.imshow(img2)
            ax3 = fig.add_subplot(222)
            ax3.imshow(xxx)
            ax4 = fig.add_subplot(223)
            ax4.imshow(ooo)
            ax=fig.add_subplot(224,projection="3d")
            win = tkinter.Tk()
            frame = tkinter.Frame(win, width=400, height=400)
            btn1 = tkinter.Button(frame, text='start', command=ini)
            btnx = tkinter.Button(frame, text='x-up', command=oriform)
            btnx2 = tkinter.Button(frame, text='x-down', command=proform)
            btnx3 = tkinter.Button(frame, text='rotate1', command=rt1)
            btnx4 = tkinter.Button(frame, text='rotate2', command=rt2)
            btnx5 = tkinter.Button(frame, text='rotate3', command=rt3)
            btnx6 = tkinter.Button(frame, text='rotate4', command=rt4)

            btn1.pack()
            canvs = FigureCanvasTkAgg(fig, win)
            canvs.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
            frame.focus_set()  # 必须获取焦点
            frame.pack()
            win.mainloop()





#########################################################################
    # for i in image_d:
    #     for j in i:
    #         print(j)
    # for i in image_d:
    #     if i.any()>2 and i.any()<255:
    #         print(i)
    # print(image_a.size,image_b.size,image_d.size)
    # # SIFT or SURF
    # img2= sift_detect(image_a, image_b,image_d)
    #
    # plt.imshow(img2)
    #
    # plt.show()