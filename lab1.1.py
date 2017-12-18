# -*- coding: utf-8 -*-
from math import *
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

# 生成标准正弦数据(x,y) 生成十个噪声数据(xa,ya)
def loadData():
    x = arange(0, 1, 0.0001)  # 画图用
    xa = arange(0, 1, 0.1)  # 随机选择10个点作为xa
    y = sin(2 * math.pi * x) #画图用的标准y
    yy = sin(2 * math.pi * xa) #随机抽取10个点的y值
    ya = []
    bia = 0.5*random.randn(len(yy), 1)  # random.randn 是生成（0,0.5）正太分布的随机函数
    for i in range(0, len(yy)):
        ya.append(yy[i] + bia[i])
    return x, y, xa, ya

# 生成x的矩阵 y的矩阵 xa的矩阵 ya的矩阵 order表示幂
def matXY(x,y,xa,ya, order):
    X = []
    Xa = []
    for i in range(order+1):  # 0-order
        X.append(x ** i)
    for i in range(order+1):
        Xa.append(xa ** i)
    X = mat(X).T  # 转置 并且形成矩阵
    Xa = mat(Xa).T
    Y = array(y).reshape((len(y), 1))  # y形成矩阵Y
    Ya = array(ya).reshape((len(ya),1))
    return X, Y,Xa,Ya

#最小二乘法的评价函数，采用十def eva_least_square(order):
    b = zeros([order+1, 1])
    for i in range(10):
        x, y, xa, ya = loadData()
        X, Y,Xa,Ya = matXY(x,y,xa,ya,order)
        XT = Xa.transpose()
        B = dot(dot(linalg.inv(dot(XT, Xa)), XT), Ya)
        b = b+B
    for i in range(order+1):
        b[i]=b[i]*0.1
    return b
def least_square(order):
    x, y, xa, ya = loadData()
    X, Y, Xa, Ya = matXY(x, y, xa, ya, order)
    XT = Xa.transpose()
    B = dot(dot(linalg.inv(dot(XT, Xa)), XT), Ya)
    finalY = dot(X, B)  # 最终算出的Y dot 矩阵乘法
    return finalY

def least_square_regular(order,l,a,ep=0.001):
 x, y, xa, ya = loadData()
 X, Y, Xa, Ya = matXY(x, y, xa, ya, order)
 ww = np.random.rand(order+1)
 w = array(ww).reshape(len(ww),1)
 while True:
    detaw = dot(dot(Xa.T,Xa),w)-dot(Xa.T,Ya)+l*w/sqrt(dot(w.T,w))
    w0 = w - a*detaw
    if linalg.norm(w-w0)< ep:
        w= w0
        break
    w = w0
 return dot(X,w)

def eva_least_square_regular(order,l,a,ep=0.001):
    b = zeros([order+1, 1])
    for i in range(10):
        x, y, xa, ya = loadData()
        X, Y, Xa, Ya = matXY(x, y, xa, ya, order)
        ww = np.random.rand(order + 1)
        w = array(ww).reshape(len(ww), 1)
        while True:
            detaw = dot(dot(Xa.T, Xa), w) - dot(Xa.T, Ya) + l * w / sqrt(dot(w.T, w))
            w0 = w - a * detaw
            if linalg.norm(w - w0) < ep:
                w = w0
                break
            w = w0
        b=b+w
    for i in range(order+1):
        b[i]=b[i]*0.1
    return b
def gradient_descent(l,order, a, m, ep=0.001, max_iter=10000):
    x, y, xa, ya = loadData()
    X, Y, Xa, Ya = matXY(x, y, xa, ya, order)  # order表示m的阶数
    w = np.random.randn(order + 1)
    w = array(w).reshape(len(w), 1)
    XTa = Xa.transpose()
    error = np.zeros(order + 1)
    count = 0  # 循环次数
    finish = 0  # 终止标志
    while count < max_iter:
        count += 1
        dif = (np.dot(Xa, w) - Ya)
        sum_m = dot(XTa, dif)
        w = w - a * sum_m + l * w / sqrt(dot(w.T, w))
        if np.linalg.norm(w - error) < ep:
            finish = 1
        else:
            error = w
    E = dot(X, w) - Y
    E2 = dot(Xa,w)-Ya
    sum = 0
    sum2 = 0
    for i in range(order):
        sum = sum + E[i] ** 2
        sum2 = sum +E2[i] **2
    sum = sqrt(sum / len(Y))
    sum2 =sqrt(sum / len(Ya))
    return dot(X, w),sum,sum2



def conjugate_gradient(order,l, ep=0.000000001, max_iter=1):
    x, y, xa, ya = loadData()
    X, Y, Xa, Ya = matXY(x, y, xa, ya, order)
   # w = np.zeros([order + 1, 1])
    w = np.random.randn(order + 1)
    w = array(w).reshape(len(w), 1)
    A = dot(Xa.T, Xa)
    r = Ya - dot(A, w)
    p = r
    sum = 0
    k = 0
    while sum < ep :
        k+=1
        a = dot(r.T, r) / dot(dot(p.T, A), p)
        w = w - dot(p, a)+l * w / sqrt(dot(w.T, w))
        r1 = r - dot(dot(A, p), a)
        p = r1 + dot(p, dot(r1.T, r1) / dot(r.T, r))
        r = r1
        for i in range(len(ya)):
            sum = sum + r1[i] ** 2
    E = dot(X, w) - Y
    E2 = dot(Xa, w) - Ya
    sum = 0
    sum2 = 0
    for i in range(order):
        sum = sum + E[i] ** 2
        sum2 = sum + E2[i] ** 2
    sum = sqrt(sum / len(Y))
    sum2 = sqrt(sum / len(Ya))
    return dot(X, w), sum, sum2

def evaluation():
    #x = arange(0, 1, 0.1)
    #y = sin(2 * math.pi * x)
    x, y, xa, ya = loadData()
    YYY = array(ya).reshape(len(ya),1)
    YY = array(y).reshape(len(y),1)
    yeva1 = zeros([18, 1])
    yeva2 = zeros([18, 1])
    yeva3 = zeros([10, 1])
    yeva4 = zeros([10, 1])
    sum = 0
    for i in range(18):
        l = [math.exp(-100),math.exp(-40),math.exp(-38),math.exp(-36),math.exp(-34),math.exp(-32),math.exp(-30),math.exp(-28)
            ,math.exp(-26),math.exp(-24),math.exp(-22),math.exp(-20),math.exp(-18),
             math.exp(-16),math.exp(-14),math.exp(-12),math.exp(-10),1]
        ls = eva_least_square_regular(9,l[i],0.03,ep=0.001)
        X = []
        Xa = []
      #  XXX, YYY, Xa, Ya = matXY(x, y, xa, ya, i)
        for o in range(9 + 1):  # 0-order
            X.append(x ** o)
            Xa.append(xa ** o)
        X = mat(X).T
        Xa = mat(Xa).T
        temp = dot(X, ls) - YY
        temp1 = dot(Xa,ls) -YYY
        sum = 0
        sum1 = 0
        for j in range(len(temp)):
            sum = sum + temp[j] ** 2
        for j in range(len(temp1)):
            sum1 = sum1 + temp1[j]**2
        yeva1[i]= sqrt(sum/len(x))
        yeva2[i]= sqrt(sum1/len(xa))
    e = [-100,-40, -38, -36, -34, -32, -30,
            -28,-26,-24,-22,-20,-18,-16,-14,
                -12,-10,0]
   # plt.plot(e, yeva1, color='b')
   # plt.plot(e,yeva2,color = 'r')

  #  for i in range(10):
  #      ls = least_square_regular(i,5e-8,0.03)
  #      X = []
  #      for i in range(i + 1):  # 0-order
  #          X.append(x ** i)
  #      X = mat(X).T
  #      temp = dot(X, ls) - YY
  #      sum = 0
  #      for j in range(len(temp)):
  #          sum = sum + temp[j] ** 2
  #      yeva2[i]= sqrt(sum/len(x))
  #  e = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  #  plt.plot(e, yeva2, color='r')



 #   lsr = least_square_regular(order, l, a)
 #   gd = gradient_descent(order, a, n)
 #   cg = conjugate_gradient(order)

def Main():
    order = 9
    a = 0.03
    b = 0.3
    l = 1e-1000000
    n = 9
    count = 10
    x, y, xa, ya = loadData()
    X, Y, Xa, Ya = matXY(x, y, xa, ya, order)

   # evaluation()
    ls = least_square(4)
    lsr = least_square_regular(order,l,a)
   #  S = []
   #  SS = []
   #  L = []
   #  l=-101
   #  for i in range(100):
   #      l +=1
   #      L.append(l)
   #      cg, sum,sum2 = conjugate_gradient(order,math.exp(l))
   #      S.append(sum)
   #      SS.append(sum2)
  #  S = array(S).reshape(len(S),1)
  #  SS = array(SS).reshape(len(SS),1)
  #  plt.plot(L,S,color = 'r')
  #  plt.plot(L,SS,color = 'b')
   # gd, sum = gradient_descent(0, order, a, n)
    cg,sum,sum2 = conjugate_gradient(order,1)
    plt.plot(x, y, color='g')  # 画图
    plt.scatter(xa,ya,color='b',marker='x')
    plt.plot(x, ls, color='r')
 #   plt.plot(x, gd, color='k')
 #   plt.plot(x, cg, color='y')
   # plt.plot(x, lsr, color='m')
    plt.show()
Main()

