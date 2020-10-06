# coding: utf-8
import json
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


def MCL(matrix,e,r):
    global qb
    global targ
    l=len(matrix)
    for i in range(l):
        matrix[i][i]=1
    yi=matrix/matrix.sum(axis=0)
    yi2=yi
    past=np.zeros((len(matrix), len(matrix)))
    sum=0
    while((past==yi2).all()==False and sum<2000):
        past=yi2
        yi2=matrix_power(yi2,e)
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
    if a[1]-b[1]==0:
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
    s1=0
    s2=0
    d1=distan(p1.pt,p2.pt)
    d2=distan(q1.pt,q2.pt)
    for i in range(len(grapy_q)):
        if(grapy_p[p1.class_id][i]<d1):
            s1=s1+1
    for i in range(len(grapy_p)):
        if(grapy_q[q1.class_id][i]<d2):
            s2=s2+1
    if(s1<20 and s2<20):
        return True
    else:
        return False
def scale(p1,q1,p2,q2):
    l1=distan(p1.pt,p2.pt)
    l2=distan(q1.pt,q2.pt)
    threshold=1.5
    if(l1!=0 and l2 !=0):
        if (1 / threshold <= ((p1.size / l1) / (q1.size / l2)) <= threshold and 1 / threshold <= (
                (p2.size / l1) / (q2.size / l2)) <= threshold):
            return True
    return False
##########################????????????????????????
def orientation(p1,q1,p2,q2):
    a=abs(ang(p1.pt,p2.pt)%180-ore(p1.pt)%180)
    b=abs(ang(q1.pt,q2.pt)%180-ore(q1.pt)%180)

    c=abs(ang(p1.pt,p2.pt)%180-ore(p2.pt)%180)
    d=abs(ang(q1.pt,q2.pt)%180-ore(q2.pt)%180)
    print("1111111111111111111", a, b, c, d)

    #if(abs(a-b)<=20 or abs(c-d)<=20):
    if abs(a - b) <= 30 and abs(c-d) <=30:
        return True
    else:
        return False


def sift_detect(img1, img2):
    global grapy_p
    global grapy_q
    sift = cv2.xfeatures2d.SIFT_create()

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
            if(neighbor(p1,q1,p2,q2)):
                if(scale(p1,q1,p2,q2)):
                    if(orientation(p1,q1,p2,q2)):
                        grapy_all[i][j]=1
                        grapy_all[j][i]=1
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
    bob=MCL(grapy_all,2,2)#get cluster
    lens=[]
    long=2
    # for i in bob:
    #     if len(i)>=2:
    #         long=3
    if(len(bob)>1):
        for j in bob:
            lens.append(len(j))
        for i in range(30):
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
    print(len(apple))
    # cv2.drawMatchesKnn expects list of lists as matches.
    if len(apple)<1:
        print("not good")
    imgg =drawMatches(img1, kp1, img2, kp2, apple)
    img2 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, (0, 0, 255), (255, 0, 0), None, flags=2)
    return bgr_rgb(imgg/255)









if __name__ == "__main__":
    # load image
    image_a = cv2.imread('/home/hexin/桌面/dataset/cool.png')
    image_b = cv2.imread('/home/hexin/桌面/dataset/bad.png')
    image_d = cv2.imread('/home/hexin/桌面/cool2.png')
    ###########################################################
    # getters and setters directly get and set on device

    #k4a.whitebalance = 4510
    #assert k4a.whitebalance == 4510
    #a = json.loads(k4a.get_calibra())
    #inner=a["CalibrationInformation"]['Cameras'][0]['Intrinsics']["ModelParameters"]


            #cv2.imwrite('/home/hexin/桌面/cool2.png', img_color)
    print(len(image_a))
            #img2,xxx,s,center= sift_detect(image_a, img_color[:, :, :3], im)
    img2 = sift_detect(image_a, image_b)



    plt.imshow(img2)


    plt.show()
