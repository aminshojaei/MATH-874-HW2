# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:20:41 2020

@author: a335s717
"""

import numpy as np

Y= np.random.normal(0, np.sqrt(1),100)

X = np.random.uniform(-1,0,[100,4])

bHat = np.dot (np.dot ( np.linalg.inv(np.dot(X.T , X)) , X.T) , Y)

