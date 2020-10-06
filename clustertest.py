from math import sqrt

import numpy as np
from numpy.linalg import matrix_power

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


def vector(p1, p2, p3):
    a = ((p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]))
    b = ((p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]))
    c = ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))
    l = sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2))
    print(a,b,c)
    if c < 0:
        c = c * (-1)
        a = a * (-1)
        b = b * (-1)
    if l != 0:
        return [a / l, b / l, c / l]
    else:
        return [a, b, c]

w=np.zeros((7,7))
ss=[[0,1/7,1,1/3,0,0,0],[1/7,0,0,1/7,0,0,0],[1,0,0,1/2,1/6,0,0],[1/3,1/7,1/2,0,0,0,0],[0,0,1/6,0,0,1/2,1/3],[0,0,0,0,1/2,0,1],[0,0,0,0,1/3,1,0]]
for i in range(7):
    for j in range(7):
        print(1)
        w[i][j]=ss[i][j]
print(w)
print(MCL(w,2,2))
# print(vector([-158, 22, 270], [-157, 26, 270], [-158, 24, 269]))
# print(int(65535))