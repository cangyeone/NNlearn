# -*- coding: utf-8 -*-
"""
All data mining method is deisgnded to get information.
Prior information can be used in many aspect.
Note that it is defferent from Prior Probability.
"""

import numpy as np
from scipy.linalg import inv
import random


def norm2(a):
    #return ||a||_2
    return np.sqrt(a.transpose().dot(a))

def supp(a,N):
    return np.argsort(a)[::-1][:N]

def re_build(y,n,k,H):
    r = y   
    j = 1  
    precision = 1e-8
    max_itr = 350
    dist = 0.0001
    T = []
    while j <= max_itr and norm2(r)/float(norm2(y)) > dist:
        err = np.abs(H.transpose().dot(r))  
        w = supp(err,2*k)       
        Omega = np.where( err[w] > precision)
        Omega = w[Omega]
        T = np.union1d(Omega,T).astype('int')  
        HT = H[:,T]
        Ht = inv(HT.transpose().dot(HT)).dot(HT.transpose())
        bT = Ht.dot(y)  

        w = supp(np.abs(bT),k)              
        updated_indices = np.where( np.abs(bT[w]) > precision)
        updated_indices = w[updated_indices]     
        T = T[updated_indices]
        x = np.zeros(H.shape[1])
        x[T] = bT[w]
        r = y - H.dot(x)        
        j +=1
    return x,r

def gen_data():
    n = 200 
    k = 30 
    m = 90

    x = np.zeros(n)
    for itr in range(k):
        x[np.random.randint(n)]=np.random.normal()
    xsparse = x.T
    H = np.random.normal(0,1./np.sqrt(m),(m,n))
    y = H.dot(xsparse)
    return y,H,x

y,H,origx=gen_data()
m=90
n=200
x,r=re_build(y,200,30,H)
print("Compressed signal is:")
print("shape:%d"%(np.shape(y)))
print(y)
#print("Gauss random Matrix is:")
#print(H)
print("rebuild signal:")
print("shape:%d"%(np.shape(x)))
print(x)