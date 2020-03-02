# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 18:16:07 2020

@author: a335s717
"""

import numpy as np
from scipy import stats


z= np.random.normal(2, np.sqrt(5),25)
T= (np.mean(z) -2)/ np.sqrt(5/25)
P_value = stats.t.cdf(T, df=24)