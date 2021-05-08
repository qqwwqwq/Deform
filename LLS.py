import math
import random
import tkinter

import numpy as np
import cv2
import tk as tk
from matplotlib import pyplot as plt
import cv2
import numpy
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
from sympy.abc import x, y, z
from result_cluster import clater

seterror = []


def check(coordinate,imagd):
    # global seterror
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    xw=coordinate[0]
    yw=coordinate[1]
    zw=coordinate[2]
    # print(coordinate,"point")
    y=int(xw*fx/zw+cx)
    x=int(yw*fy/zw+cy)
    if x>3071 or y>4095 or x<0 or y<0:
        return 100000000
    z=imagd[x][y]
    # print(x, y,  z,xw,yw,zw, 'xy')
    # print(z)
    xx=  (y - cx) * z / fx
    yy= (x - cy) * z / fy
    print(xx, yy, z, xw, yw, zw, 'xy')
    #
    print("ffffffffffffffffffffffffk",pow(zw-z,2))
    errors=sqrt(pow(xw-xx,2)+pow(yw-yy,2)+pow(zw-z,2))
    # errors = abs(xw - xx) * abs(yw - yy) * abs(zw - z)
    # errors=abs(zw-z)
    print(errors, "erors")
    # print(xx,yy,errors,"errors")
    return errors
    # if errors<1:
    #     return True
    # else:
    #     return False
def check2(coordinate,imagd,plane):
    # global seterror
    dir=plane.direct
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    xw=coordinate[0]
    yw=coordinate[1]
    zw=coordinate[2]
    # print(coordinate,"point")
    y=int(xw*fx/zw+cx)
    x=int(yw*fy/zw+cy)
    if x>3071 or y>4095 or x<0 or y<0:
        return 100000000
    z=imagd[x][y]
    # print(x, y,  z,xw,yw,zw, 'xy')
    # print(z)
    xx=  (y - cx) * z / fx
    yy= (x - cy) * z / fy
    print(xx, yy, z, xw, yw, zw, 'xy')
    #
    errors=abs(xx*dir[0]+yy*dir[1]+z*dir[2]+plane.d)
    # errors=sqrt(pow(xw-xx,2)+pow(yw-yy,2)+pow(zw-z,2))
    # errors = abs(xw - xx) * abs(yw - yy) * abs(zw - z)
    # errors=abs(zw-z)
    print(errors, "erors")
    # print(xx,yy,errors,"errors")
    return errors
    # if errors<1:
    #     return True
    # else:
    #     return False
def bending(c,d):
    a=c.direct
    ap = c.locate
    b=d.direct
    # print(a,b,"two")
    bp = d.locate
    # da=np.dot(np.array(a),np.array(ap))
    # db=np.dot(np.array(b),np.array(bp))
    da=c.d*(-1)
    db=d.d*(-1)
    x1 = a[1] * b[2] - b[1] * a[2]
    y2 = a[2] * b[0] - a[0] * b[2]
    z3 = a[0] * b[1] - a[1] * b[0]
    t = [x1, y2, z3]
    #print(da,db,a,b,t,"ddd")
    px=1
    aa = solve([px*a[0] + y*a[1] +z*a[2]- da, px * b[0] + b[1] * y +z*b[2]- db], [y, z])
    #print(aa.keys())
    gap=[1]
    if y in aa.keys():
        #print("111")
        gap.append(aa[y])
    else:
        gap.append(0)
    if z in aa.keys():
        gap.append(aa[z])
    else:
        gap.append(0)
    #print(gap)
    return np.array(t),np.array(gap)
#t is the direction, gap is the
def slide(t,gap,start,end,planea,planeb,imgd,flag):
    alpa =3
    der = sqrt(pow(t[0], 2) + pow(t[1], 2) + pow(t[2], 2))
    if flag==0:
        ders=int(alpa*abs(t[0])/der)
        if ders==0:
            ders=1
        print(der,ders)
    else:
        ders = int(alpa* abs(t[1]) / der)
        if ders==0:
            ders=1
        print(der, ders)
    gapa=planea.direct
    gapb=planeb.direct
    # da=np.dot(np.array(planea.direct),np.array(planea.locate))
    # db=np.dot(np.array(planeb.direct),np.array(planeb.locate))
    da=planea.d*(-1)
    db=planeb.d*(-1)
    def cal(point):
        state=(point-gap[0])/t[0]
        y=state*t[1]+gap[1]
        z=state*t[2]+gap[2]
        return [point,y,z]
    def cal2(point):
        state=(point-gap[1])/t[1]
        x=state*t[0]+gap[0]
        z=state*t[2]+gap[2]
        return [x,point,z]
    print(start, end-alpa)
    sum=[0]*4
    tree=10
    step=end-start-80
    time=[0]*4
    switch=[0]*4
    for i in range(start,end-alpa,ders):

        print(1)
        if flag==0:
            point1=cal(i)
            point2=cal(i+ders)
        else:
            point1 = cal2(i)
            point2 = cal2(i + ders)
        if max(sum) >(step/ders)/2:
            sob = i
            qob = i+der
            if flag==0:
                s1=cal(sob)
                s2=cal(qob)
            else:
                s1=cal2(sob)
                s2=cal2(qob)
            print(check(s1,imgd),check(s2,imgd))
            # print(point1, point2)
            while check(s1,imgd)<15:
                if flag==0:
                    sob-=8
                    s1=cal(sob)
                else:
                    sob -= 8
                    s1 = cal2(sob)
            while check(s2,imgd)<15:
                if flag==0:
                    qob+=8
                    s2=cal(qob)
                else:
                    qob += 8
                    s2 = cal2(qob)
            drawlines(s1, s2, imgd)
            # drawlines(point1, point2, imgd)
            return True
        # print("p1p2-------------------------------",check(point1, imgd),check(point2, imgd),point1,point2)
        if check(point1, imgd)<tree+2 :
            if check(point2, imgd)<tree+2 :
                cc = solve([gapa[0] * x + gapa[1] * y + gapa[2] * z - da,
                            (x - point1[0]) ** 2 + (y - point1[1]) ** 2 + (z - point1[2]) ** 2 - 25,
                            t[0] * (point1[0] - x) + t[1] * (point1[1] - y) + t[2] * (point1[2] - z)], [x, y, z])
                dd = solve([gapb[0] * x + gapb[1] * y + gapb[2] * z - db,
                            (x - point2[0]) ** 2 + (y - point2[1]) ** 2 + (z - point2[2]) ** 2 -25,
                            t[0] * (point2[0] - x) + t[1] * (point2[1] - y) + t[2] * (point2[2] - z)], [x, y, z])
                # print("ccdd", cc, dd,point1,point2)
                # p1,p2=check2(cc[0], imgd,planea),check2(cc[1], imgd,planea)
                # q1,q2=check2(dd[0], imgd,planeb), check2(dd[1], imgd,planeb)
                p1, p2 = check(cc[0], imgd), check(cc[1], imgd)
                q1, q2 = check(dd[0], imgd), check(dd[1], imgd)

                print([p1,p2,q1,q2],"1234")
                print(sum,"sum")
                if p1<tree and q1<tree :
                    if sum[0]==time[0]:
                        sum[0]+=1
                        switch[0]=0
                        print(1111111)
                    else:
                        print(-1)
                        sum[0]=0
                        time[0]=0
                        switch[0]=1

                if p1<tree and q2<tree :

                    if sum[1]==time[1]:
                        switch[1]=0
                        sum[1]+=1
                        print(222222)
                    else:
                        print(-2)
                        sum[1]=0
                        switch[1]=1
                        time[1]=0


                if p2<tree and q1<tree :


                    if sum[2] == time[2]:
                        switch[2] = 0
                        sum[2] += 1
                        print(333333)
                    else:
                        print(-3)
                        sum[2] = 0
                        switch[2] = 1
                        time[2] = 0
                if p2<tree and q2<tree :
                    if sum[3] == time[3]:
                        switch[3] = 0
                        sum[3] += 1
                        print(44444)
                    else:
                        print(-4)
                        sum[3] = 0
                        switch[3] = 1
                        time[3] = 0
                for i in range(len(switch)):
                    if switch[i]==0:
                        time[i]+=1
                if (p2>tree and p1>tree) or (q1>tree and q2>tree):
                    sum=[0]*4
                    time=[0]*4
                    switch=[0]*4
            else:
                sum=[0]*4
                time = [0] * 4
                switch = [0] * 4
        else:
            sum=[0]*4
            time = [0] * 4
            switch = [0] * 4

    return False



def LLs(x, y, z, size):
    a = 0
    A = np.ones((size, 3))
    for i in range(0, size):
        A[(i, 0)] = x[a]
        A[(i, 1)] = y[a]
        a = a + 1

    b = np.zeros((size, 1))
    a = 0
    for i in range(0, size):
        b[(i, 0)] = z[a]
        a = a + 1

    A_T = A.T
    A1 = np.dot(A_T, A)
    A2 = np.linalg.inv(A1)
    A3 = np.dot(A2, A_T)
    X = np.dot(A3, b)
    for i in X:
        i[0] = float(i[0])

    return X


def drawlines(p1, p2, imgd):
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    xw1 = p1[0]
    yw1 = p1[1]
    zw1 = p1[2]
    xw2 = p2[0]
    yw2 = p2[1]
    zw2 = p2[2]
    y1 = xw1 * fx / zw1 + cx
    x1 = yw1 * fy / zw1 + cy
    y2 = xw2 * fx / zw2 + cx
    x2 = yw2 * fy / zw2 + cy
    # x3 = x1
    # x4 = x2
    # y3 = (x3 - x1) * (y2 - y1) / (x2 - x1) + y1
    # y4 = (x4 - x1) * (y2 - y1) / (x2 - x1) + y1
    cv2.line(imgd, (int(y1), int(x1)), (int(y2), int(x2)), (0, 0, 255), thickness=16)


def angelsurface(a, b):
    cosang = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / sqrt(pow(a[0], 2) + pow(a[1], 2) + pow(a[2], 2)) * sqrt(
        pow(b[0], 2) + pow(b[1], 2) + pow(b[2], 2))
    angofsur = np.math.acos(round(cosang, 3))
    res = angofsur * 180 / np.pi
    return round(res, 3)


def growprocess(seed, imgd):
    img=imgd.copy()
    up = seed[0][0]
    down = seed[0][0]
    left = seed[0][1]
    right = seed[0][1]
    print('start')
    setx = []
    sety = []
    setz = []

    def check(point=None, tre=None):
        cx = 2044.08
        cy = 1550.39
        fx = 1955.83
        fy = 1955.42
        x = point[0]
        y = point[1]
        z = imgd[x][y]
        xw1 = int((y - cx) * z / fx)
        yw1 = int((x - cy) * z / fy)
        res = abs(dir[0] * xw1 + dir[1] * yw1 + dir[2] * z + distan)
        if res - tre < 0:
            return True

    def rr(tw=None):
        cx = 2044.08
        cy = 1550.39
        fx = 1955.83
        fy = 1955.42
        x = tw[0]
        y = tw[1]
        z = imgd[x][y]
        xw1 = (y - cx) * z / fx
        yw1 = (x - cy) * z / fy
        (setx.append(xw1), sety.append(yw1), setz.append(z))

    # def check2(a, b, flag):
    #     def dis(first,second,third):
    #         cx = 2044.08
    #         cy = 1550.39
    #         fx = 1955.83
    #         fy = 1955.42
    #         x1=(first[1] - cx) * first[2] / fx
    #         y1=(first[0] - cy) * first[2] / fy
    #         x2=(second[1] - cx) * second[2] / fx
    #         y2=(second[0] - cy) * second[2] / fy
    #         x3 = (third[1] - cx) * third[2] / fx
    #         y3 = (third[0] - cy) * third[2] / fy
    #         d1=[int(x1-x2),int(y1-y2),float(first[2])-float(second[2])]
    #         d2=[int(x2-x3),int(y2-y3),float(second[2])-float(third[2])]
    #         print(d1,d2)
    #         cosang = (d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2]) / (sqrt(pow(d1[0], 2) + pow(d1[1], 2) + pow(d1[2], 2)) * sqrt(
    #         pow(d2[0], 2) + pow(d2[1], 2) + pow(d2[2], 2)))
    #         angofsur = math.acos(round(cosang, 3))
    #         res = angofsur * 180 / np.pi
    #         print(res)
    #         return res
    #     if flag == 1:
    #         print(1)
    #         ii = b
    #
    #         while imgd[a][b] == imgd[a][ii] and i < seed[0][1]:
    #             ii += 1
    #         jj = ii
    #         while imgd[a][ii] == imgd[a][jj] and jj < seed[0][1]:
    #             jj += 1
    #         if b!=ii and ii!=jj and dis([a,b,float(imgd[a][b])],[a,ii,float(imgd[a,ii])],[a,jj,float(imgd[a,jj])])<30:
    #             return True
    #         return False
    #     if flag == 2:
    #         print(2)
    #         ii = b
    #         while imgd[a][b] == imgd[a][ii] and ii > seed[0][1]:
    #             ii -= 1
    #         jj = ii
    #         while imgd[a][ii] == imgd[a][jj] and jj > seed[0][1]:
    #             jj -= 1
    #         if  b!=ii and ii!=jj and dis([a, b, imgd[a][b]], [a, ii, imgd[a, ii]], [a, jj, imgd[a, jj]]) <30:
    #             return True
    #         return False
    #     if flag == 3:
    #         print(3)
    #         ii = a
    #         while imgd[a][b] == imgd[ii][b] and ii > seed[0][0]:
    #             ii -= 1
    #         jj = ii
    #         while imgd[ii][b] == imgd[jj][b] and jj > seed[0][0]:
    #             jj -= 1
    #         if  dis([a, b, imgd[a][b]], [ii, b, imgd[ii, b]], [jj, b, imgd[jj, b]])<30:
    #             return True
    #         return False
    #     if flag == 4:
    #         print(4)
    #         ii = a
    #         while imgd[a][b] == imgd[ii][b] and ii < seed[0][0]:
    #             ii += 1
    #         jj = ii
    #         while imgd[ii][b] == imgd[jj][b] and jj < seed[0][0]:
    #             jj += 1
    #         if  a!=ii and ii!=jj and dis([a, b, imgd[a][b]], [ii, b, imgd[ii, b]], [jj, b, imgd[jj, b]]) < 30:
    #             return True
    #         return False

    def check2(a,b):
        o1=seed[0][0]
        o2=seed[0][1]
        cx = 2044.08
        cy = 1550.39
        fx = 1955.83
        fy = 1955.42
        x1=(b - cx) * imgd[a][b] / fx
        y1=(a - cy) * imgd[a][b]/ fy
        x2=(o2 - cx) * imgd[o1][o2] / fx
        y2=(o1 - cy) * imgd[o1][o2] / fy
        out=[(x1+x2)/2,(y1+y2)/2,(float(imgd[a][b])+float(imgd[o1][o2] ))/2]
        yy=int(out[0]*fx/out[2]+cx)
        xx=int(out[1]*fy/out[2]+cy)
        zz=imgd[xx][yy]
        xw=(yy - cx) * zz / fx
        yw=(xx-cy)*zz/fy
        #print(out,[xw,yw,zz])
        error=sqrt(pow(out[0]-xw,2)+pow(out[1]-yw,2)+pow(float(out[2])-float(zz),2))
        #print(error)
        return error








    dir = []
    distan = 0
    j = 1
    thre = 0
    rr(seed[0])
    preve = []
    trge = 3
    while j < 150:
        # print(dir, distan)
        if j ==1:
            seed0 = []
            for i in range(down - 1, up + 2):
                if imgd[(i, left - 1)] != 0:
                    seed0.append([i, left - 1])
                if imgd[(i, right + 1)] != 0:
                    seed0.append([ i,right + 1])
            for i in range(left, right + 1):
                if imgd[(up + 1, i)] != 0:
                    seed0.append([up + 1,i])
                if imgd[(down - 1, i)] != 0:
                    seed0.append([ down - 1,i])
            for i in seed0:
                rr(i)

            preve = seed0[:]
            seed.extend(seed0)
            if up + 1 < 3072:
                up += 1
            if down > 1:
                down -= 1
            if left > 1:
                left -= 1
            if right < 4096:
                right += 1
            A = LLs(setx, sety, setz, len(setz))
            dir = [  A[0][0],   A[1][0],  -1]
            distan = A[2][0]
            j += 1
            continue
        seed1 = []
        print(j)
        ober=2.7
        if j < 3072*4096/400:
            thre = 4* pow(1 - pow(numpy.e, -j), 2)
        else:
            thre = 0.09 * pow(distan, 2) * 9 * pow(1 - pow(np.e, -j), 2)
        for i in range(down - 1, up + 2):
            if  imgd[(i, left - 1)] != 0 :
                if  [i,  left] in preve or [ i - 1, left] in preve or [i + 1,  left] in preve :
                    if check2(i,left-1)<ober:
                        seed1.append([ i, left - 1])
        for i in range(down - 1, up + 2):
            if imgd[(i, right + 1)] != 0:
                if [i, right] in preve or [i - 1, right] in preve or [i + 1, right] in preve :
                    if check2(i, right+1)<ober:
                        seed1.append([i,right + 1])
        for i in range(left, right + 1):
            if imgd[(up + 1, i)] != 0:
                if [up, i] in preve or [up,i - 1] in preve or [up,i + 1] in preve :
                    if check2(up+1, i)<ober:
                        seed1.append([up + 1, i])
        for i in range(left, right + 1):
            if imgd[(down - 1, i)] != 0:
                if [down, i] in preve or [down, i - 1] in preve or [down, i + 1] in preve :
                    if check2(down-1, i)<ober:
                        seed1.append([down - 1, i])
        preve = seed1[:]
        for i in seed1:
            rr(i)
        seed.extend(seed1)
        if up < 3071:
            up += 1
        if down > 1:
            down -= 1
        if left > 1:
            left -= 1
        if right < 4096:
            right += 1
        A = LLs(setx, sety, setz, len(setz))
        dir = [A[0][0],A[1][0],-1]
        distan = A[2][0]
        j += 1
    #
    for i in seed:
        img[i[0]][i[1]]=0
    plt.imshow(img)
    plt.show()
    return (max(setx), min(setx), dir, distan, max(sety), min(sety))

def cldis(a,b):
    dira=a.direct
    da=a.d
    cena=a.locate
    dirb=b.direct
    db=b.d
    cenb=b.locate
    error1=abs(cenb[0]*dira[0]+cenb[1]*dira[1]+cenb[2]*dira[2]+a.d)
    error2=abs(cena[0]*dirb[0]+cena[1]*dirb[1]+cena[2]*dirb[2]+b.d)
    print(error1,error2,"12")
    return abs(error2+error1)
def cret(s, img):
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    z1 = img[s[0][0]][s[0][1]]
    xw1 = (s[0][1] - cx) * z1 / fx
    yw1 = (s[0][0] - cy) * z1 / fy
    (w1, w2, w3, w4, w5, w6) = growprocess(s, img)
    out1 = clater(img, w3, [
        xw1,
        yw1,
        z1], 0, w4, w1, w2, s[0], w5, w6)
    return out1
#
img = np.load('/home/hexin/桌面/dep.npy')
out1=cret([[1650,1300]],img)
out2=cret([[1600,1900]],img)
print(cldis(out1,out2))
# print(out1.direct,out1.d,out2.direct,out2.d,"two plane")
# t,gap=bending(out1,out2)
# print("bending",t,gap)
# if abs(t[0])<abs(t[1]):
#     print("f1")
#     if slide(t, gap, int(min(out1.miny, out2.miny)) - 40,
#                             int(max(out1.maxy, out2.maxy)) + 40, out1, out2, img,1):
#
#         plt.imshow(img)
#         plt.show()
# else:
#     if slide(t, gap, int(min(out1.minx, out2.minx)) - 40,
#             int(max(out1.maxx, out2.maxx)) + 40, out1, out2, img, 0):
#         plt.imshow(img)
#         plt.show()