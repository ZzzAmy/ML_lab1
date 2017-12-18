# -*- coding: utf-8 -*-
from math import *
import matplotlib.pyplot as plt
from numpy import *
import numpy as np

def loadData():
    dataMat = [];labelMat = []
    fr = open('data.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        print(lineArr)
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat
def testData():
    dataMat = [];
    labelMat = []
    fr = open('test.txt')
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
        dataMat1=array(dataMat)
        x = array(dataMat1[:,1])
    return dataMat, labelMat,x
def loadData_2wei():
    data = []
    label = []
    x = arange(-1, 1, 0.02)  # 画图用
    y = 7*x  #画图用的标准y
    bia = 8*random.randn(len(y),1)
    xe = arange(-1,1,0.02)
    j=-5
    for i in range(0, len(y)):
        data.append([1.0,float(x[i]),float(y[i]+bia[i])])
        if(y[i]+bia[i] <= y[i]):
            if(x[i] == xe[j]):
                label.append(1)
                j = j+5
            else:
                label.append(0)
        else:
            if(x[i]==xe[j]):
                label.append(0)
                j = j+5
            else:
                label.append(1)
    return data,label
def loadData_nwei():
    data = []
    label = []
    x = arange(-1, 1, 0.02)  # 画图用
    y = 5 * x  # 画图用的标准y
    bia = 5 * random.randn(len(y), 1)
    xe = arange(-1, 1, 0.02)
    j = -5
    for i in range(0, len(y)):
        data.append([1.0, float(x[i]), float(y[i] + bia[i]),
                     float(x[i]*(y[i]+bia[i])),float(x[i]*x[i]),float(y[i]*y[i]) ] )
        if (y[i] + bia[i] <= y[i]):
            if (x[i] == xe[j]):
                label.append(1)
                j = j + 5
            else:
                label.append(0)
        else:
            if (x[i] == xe[j]):
                label.append(0)
                j = j + 5
            else:
                label.append(1)
    return data, label
def loadData_test_nwei():
    data = []
    label = []
    x = arange(-1,1,0.1)
    y = x*5
    bia = 5 * random.randn(len(y), 1)
    xe = arange(-1,1,0.1)
    j = -5
    for i in range(0, len(y)):
        data.append([1.0, float(x[i]), float(y[i] + bia[i]),float(x[i]*(y[i]+bia[i])),float(x[i]*x[i])
                        ,float(y[i]*y[i])])
        if (y[i] + bia[i] <= y[i]):
            if (x[i] == xe[j]):
                label.append(1)
                j = j + 5
            else:
                label.append(0)
        else:
            if (x[i] == xe[j]):
                label.append(0)
                j = j + 10
            else:
                label.append(1)
    return data, label,x

def loadData_test_2wei():
    data = []
    label = []
    x = arange(-1, 1, 0.02)  # 画图用
    y = 5 * x  # 画图用的标准y
    bia = 5 * random.randn(len(y), 1)
    xe = arange(-1, 1, 0.02)
    j = -5
    for i in range(0, len(y)):
        data.append([1.0, float(x[i]), float(y[i] + bia[i])])
        if (y[i] + bia[i] <= y[i]):
            if (x[i] == xe[j]):
                label.append(1)
                j = j + 5
            else:
                label.append(0)
        else:
            if (x[i] == xe[j]):
                label.append(0)
                j = j + 5
            else:
                label.append(1)
    return data, label,x
def sigmoid(X):
    return 1.0/(1+exp(-X))
#求梯度
def gradient(dataMat,labelMat,w):
    h = sigmoid(dataMat * w)
    error = labelMat - h
    temp = dataMat.transpose()*error
    return temp
#求海赛矩阵
def HesseMat(dataMat,w):
    h = sigmoid(dataMat * w)
    hMat = mat(h)
    m,n = shape(dataMat)
    ht = ones((m,1))
    htMat = ht - h
    HI = eye(m)
    for i in range(n):
        HI[i,i]=hMat[i]*htMat[i]
    n = dataMat.transpose()*HI*dataMat
    return n

def Neton(data,label,ep,max_iter):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    m,n = shape(dataMat)
    w = ones((n,1))
    time = 0
    for i in range(max_iter):
        time = time+1
        if(time%1000 == 0):
            print(time)
            print(linalg.norm(G))
        H = HesseMat(dataMat,w)
        if(linalg.det(H) == 0):
            w = zeros((n,1))
            break
        else:
            G = gradient(dataMat, labelMat, w)
            if (linalg.norm(G) < ep):
                break
            else:
                w = w + dot(H.I, G)
    print("newton")
    print(time)
    print(linalg.norm(G))
    return w
def Neton_regularization(data,label,ep,max_iter,l=0.01):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    m,n = shape(dataMat)
    w = ones((n,1))
    e = eye(n)
    time = 0
    for i in range(max_iter):
        time = time+1
        H = HesseMat(dataMat,w) + l*e
        G = gradient(dataMat,labelMat,w) + l * w
       # if (time % 1000 == 0):
       #     print(linalg.norm(G))
       #     print(time)
        if (linalg.norm(G) < ep):
            break
        else:
            w = w + dot(H.I, G)
    print("newton_regularization")
    print(time)
    print(linalg.norm(G))
    return w

def gradDecent(data,label,alpha,ep,max_iter):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    m,n = shape(dataMat)
    w = ones((n,1))
    w1 = w
    time = 0
    for i in range(max_iter):
        time= time +1
        temp = -gradient(dataMat,labelMat,w)
        if(linalg.norm(temp) < ep ):
            break
        else:
            w = w - alpha * temp
    print("gradient")
    print(time)
    print(linalg.norm(temp))
    return w
def gradient_regularization(data,label,alpha,ep,max_iter,l=0.000001):
    dataMat = mat(data)
    labelMat = mat(label).transpose()
    m,n = shape(dataMat)
    w = ones((n,1))
    time = 0
    for i in range(max_iter):
        time = time+1
        if(time%1000 == 0):
            print(time)
            print(linalg.norm(temp))
        temp = gradient(dataMat,labelMat,w)
        if(linalg.norm(temp) < ep):
            break
        else:
             w = w + alpha * temp - l * w
    print("gradient_regularization")
    print(time)
    print(linalg.norm(temp))
    return w
def accuracy(w,x,label):
    y = sigmoid(x*w)
    no = 0
    for i in range(len(y)):
        if(y[i] > 0.5):
            if(label[i] != 1):
                no = no+1
        else:
            if(label[i]!= 0):
                no = no+1
    return 1-float(no/len(y))
def paint_2wei():
    alpha1 = 0.0001  # 步长
    ep = 0.005
    max_iter = 10000000
    max_iter1 = 500000
    data, label = loadData_2wei() #100点
    w = gradDecent(data, label, alpha1, ep, max_iter)
    print("=======================================0")
    ww = gradient_regularization(data, label, alpha1, ep, max_iter)
    print("=======================================1")
    w2 = Neton(data, label, ep, max_iter1)
    print("=========================================2")
    ww2 = Neton_regularization(data, label, ep, max_iter1)
    while (linalg.norm(w2) == 0):
        data, label = loadData_2wei()
        print("=======================================3")
        w = gradDecent(data, label, alpha1, ep, max_iter)
        print("=======================================4")
        ww = gradient_regularization(data, label, alpha1, ep, max_iter)
        print("=======================================5")
        w2 = Neton(data, label, ep, max_iter1)
        print("=======================================6")
        ww2 = Neton_regularization(data, label, ep, max_iter1)
    xx,ylabel,x = loadData_test_2wei()
    dataArr = array(xx)
    n = shape(dataArr)[0]
    x1 = [];
    y1 = []
    x2 = [];
    y2 = []
    for i in range(n):
        if label[i] == 1:
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=5, c="red", marker='s')
    ax.scatter(x2, y2, s=5, c="green")
    # 梯度下降
    a1 = accuracy(w, xx, ylabel)
    # 正则梯度下降
    aa1 = accuracy(ww, xx, ylabel)
    # 牛顿
    b1 = accuracy(w2, xx, ylabel)
    # 正则牛顿
    bb1 = accuracy(ww2, xx, ylabel)
    print("梯度下降准确率：")
    print(a1)
    print("正则梯度下降准确率：")
    print(aa1)
    print("牛顿准确率：")
    print(b1)
    print("正则牛顿准确率：")
    print(bb1)
    p1 = w.getA()
    p2 = w2.getA()
    pp1 = ww.getA()
    pp2 = ww2.getA()
    y = (-p1[0] - p1[1] * x) / p1[2]
    yy = (-pp1[0] - pp1[1] * x) / pp1[2]
    y2 = (-p2[0] - p2[1] * x) / p2[2]
    yy2 = (-pp2[0] - pp2[1] * x) / pp2[2]
    g1, = ax.plot(x, y, c="blue", label="gradient")
    n1, = ax.plot(x, y2, c="yellow", label="newton")
    g2, = ax.plot(x, yy, c="black", label="gradient_regularization")
    n2, = ax.plot(x, yy2, c='m', label="newton_regulatizatoin")
    plt.legend(handles=[g1, g2, n1, n2])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
def paint_nwei():
    alpha1 = 0.0001  # 步长
    ep = 0.005
    max_iter = 8000000
    max_iter1 = 500000
    data, label = loadData_nwei()
    w = gradDecent(data, label, alpha1, ep, max_iter)
    print("=======================================0")
    ww = gradient_regularization(data, label, alpha1, ep, max_iter)
    print("=======================================1")
    w2 = Neton(data, label, ep, max_iter1)
    print("=========================================2")
    ww2 = Neton_regularization(data, label, ep, max_iter1)
    while (linalg.norm(w2) == 0):
        data, label = loadData_nwei()
        print("=======================================3")
        w = gradDecent(data, label, alpha1, ep, max_iter)
        print("=======================================4")
        ww = gradient_regularization(data, label, alpha1, ep, max_iter)
        print("=======================================5")
        w2 = Neton(data, label, ep, max_iter1)
        print("=======================================6")
        ww2 = Neton_regularization(data, label, ep, max_iter1)
    xx, ylabel, x = loadData_test_nwei()
    dataArr = array(xx)
    n = shape(dataArr)[0]
    x1 = [];
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    x2 = [];
    y21 = []
    y22 = []
    y23 = []
    y24 = []
    for i in range(n):
        if ylabel[i] == 1:
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
            y2.append(dataArr[i, 3])
            y3.append(dataArr[i, 4])
            y4.append(dataArr[i, 5])
        else:
            x2.append(dataArr[i, 1])
            y21.append(dataArr[i, 2])
            y22.append(dataArr[i, 3])
            y23.append(dataArr[i, 4])
            y24.append(dataArr[i, 5])
    fig = plt.figure()
    ax = fig.add_subplot(111)
 #   ax.scatter(x1, y1, s=5, c="red", marker='s')
 #   ax.scatter(x2, y21, s=5, c="green")

    ax.scatter(x1, y2, s=5, c="red", marker='s')
    ax.scatter(x2, y22, s=5, c="green")

  #  ax.scatter(x1, y3, s=5, c="red", marker='s')
  #  ax.scatter(x2, y23, s=5, c="green")

#    ax.scatter(x1, y4, s=5, c="red", marker='s')
#    ax.scatter(x2, y24, s=5, c="green")
    #梯度下降
    a1 = accuracy(w,xx,ylabel)
    #正则梯度下降
    aa1 = accuracy(ww,xx,ylabel)
    #牛顿
    b1 = accuracy(w2,xx,ylabel)
    #正则牛顿
    bb1 = accuracy(ww2,xx,ylabel)
    print("梯度下降准确率：")
    print(a1)
    print("正则梯度下降准确率：")
    print(aa1)
    print("牛顿准确率：")
    print(b1)
    print("正则牛顿准确率：")
    print(bb1)

 #   p1 = w.getA()
 #   p2 = w2.getA()
 #   pp1 = ww.getA()
 #   pp2 = ww2.getA()
 #   y = (-p1[0] - p1[1] * x) / p1[2]
 #   yy = (-pp1[0] - pp1[1] * x) / pp1[2]
 #   y2 = (-p2[0] - p2[1] * x) / p2[2]
 #   yy2 = (-pp2[0] - pp2[1] * x) / pp2[2]
 #   g1, = ax.plot(x, y, c="blue", label="gradient")
 #   n1, = ax.plot(x, y2, c="yellow", label="newton")
 #   g2, = ax.plot(x, yy, c="black", label="gradient_regularization")
 #   n2, = ax.plot(x, yy2, c='m', label="newton_regulatizatoin")
 #   plt.legend(handles=[g1, g2, n1, n2])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
def paint_data():
    alpha1 = 0.0001  # 步长
    ep = 0.005
    max_iter = 80000
    max_iter1 = 500000
    data, label = loadData()
    w = gradDecent(data, label, alpha1, ep, max_iter)
    print("=======================================0")
    ww = gradient_regularization(data, label, alpha1, ep, max_iter)
    print("=======================================1")
    w2 = Neton(data, label, ep, max_iter1)
    print("=========================================2")
    ww2 = Neton_regularization(data, label, ep, max_iter1)
    while (linalg.norm(w2) == 0):
        data, label = loadData()
        print("=======================================3")
        w = gradDecent(data, label, alpha1, ep, max_iter)
        print("=======================================4")
        ww = gradient_regularization(data, label, alpha1, ep, max_iter)
        print("=======================================5")
        w2 = Neton(data, label, ep, max_iter1)
        print("=======================================6")
        ww2 = Neton_regularization(data, label, ep, max_iter1)
    xx, ylabel, x = testData()
    dataArr = array(xx)
    n = shape(dataArr)[0]
    x1 = [];
    y1 = []
    x2 = [];
    y2 = []
    for i in range(n):
        if label[i] == 1:
            x1.append(dataArr[i, 1])
            y1.append(dataArr[i, 2])
        else:
            x2.append(dataArr[i, 1])
            y2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=5, c="red", marker='s')
    ax.scatter(x2, y2, s=5, c="green")
    # 梯度下降
    a1 = accuracy(w, xx, ylabel)
    # 正则梯度下降
    aa1 = accuracy(ww, xx, ylabel)
    # 牛顿
    b1 = accuracy(w2, xx, ylabel)
    # 正则牛顿
    bb1 = accuracy(ww2, xx, ylabel)
    print("梯度下降准确率：")
    print(a1)
    print("正则梯度下降准确率：")
    print(aa1)
    print("牛顿准确率：")
    print(b1)
    print("正则牛顿准确率：")
    print(bb1)
    p1 = w.getA()
    p2 = w2.getA()
    pp1 = ww.getA()
    pp2 = ww2.getA()
    hy = (-p1[0] - p1[1] * x) / p1[2]
    hyy = (-pp1[0] - pp1[1] * x) / pp1[2]
    hy2 = (-p2[0] - p2[1] * x) / p2[2]
    hyy2 = (-pp2[0] - pp2[1] * x) / pp2[2]
    g1, = ax.plot(x, hy, c="blue", label="gradient")
    n1, = ax.plot(x, hy2, c="yellow", label="newton")
    g2, = ax.plot(x, hyy, c="black", label="gradient_regularization")
    n2, = ax.plot(x, hyy2, c='m', label="newton_regulatizatoin")
    plt.legend(handles=[g1, g2, n1, n2])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



def Main():
 #   paint_2wei()
 #   paint_nwei()
     paint_data()
Main()
