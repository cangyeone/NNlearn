# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 19:39:10 2017

@author: Cangye
"""
import numpy as np
eta=0.3
def gennext(data):
    return data+np.array([-2*(-0.5+data[0]),-2*(-0.5+data[1])])*eta
buf=[0.,0.]
for _ in range(100):
    print(buf)
    buf=gennext(buf)
    
