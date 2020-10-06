from sympy import *
from sympy.abc import x,y,z,a,b,c
from result_cluster import clater
import numpy as np
c=clater(None,[0,0,1],[1,1,1],None)
d=clater(None,[1,-1,0],[1,1,0],None)
def check(coordinate,imagd):
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    xw=coordinate[0]
    yw=coordinate[1]
    zw=coordinate[2]
    x=int(xw*fx/zw+cx)
    y=int(yw*fy/zw+cy)
    z=imagd[x][y]
    return abs(zw-z)
def bending(c,d):
    a=c.direct
    ap = c.locate
    b=d.direct
    bp = d.locate
    da=np.dot(np.array(a),np.array(ap))
    db=np.dot(np.array(b),np.array(bp))
    x1 = a[1] * b[2] - b[1] * a[2]
    y2 = a[2] * b[0] - a[0] * b[2]
    z3 = a[0] * b[1] - a[1] * b[0]
    t = [x1, y2, z3]
    print(da,db,a,b,t,"ddd")
    px=ap[0]
    aa = solve([px*a[0] + y*a[1] +z*a[2]- da, px * b[0] + b[1] * y +z*b[2]- db], [y, z])
    print(aa.keys())
    gap=[px]
    if y in aa.keys():
        print("111")
        gap.append(aa[y])
    else:
        gap.append(0)
    if z in aa.keys():
        gap.append(aa[z])
    else:
        gap.append(0)
    print(gap)
    return np.array(t),np.array(gap)
t,gap=bending(c,d)
#t is the direction, gap is the
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

# on the plane, distance, vertical

        cc = solve([gapa[0] * x + gapa[1] * y + gapa[2] * z - da,
                    (x - point1[0]) ** 2 + (y - point1[1]) ** 2 + (z - point1[2]) ** 2 - 4,
                    t[0] * (point1[0] - x) + t[1] * (point1[1] - y) + t[2] * (point1[2] - z)], [x, y, z])
        dd = solve([gapb[0] * x + gapb[1] * y + gapb[2] * z - db,
                    (x - point2[0]) ** 2 + (y - point2[1]) ** 2 + (z - point2[2]) ** 2 - 4,
                    t[0] * (point2[0] - x) + t[1] * (point2[1] - y) + t[2] * (point2[2] - z)], [x, y, z])
        print(i,cc[0],dd[0])
slide(t,gap,0,5,c,d)