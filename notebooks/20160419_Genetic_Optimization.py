
# coding: utf-8

# ##### IPython notebook setup

# In[1]:

# Import path to source directory (bit of a hack in IPython)
import sys
import os
pwd = get_ipython().magic('pwd')
sys.path.append(os.path.join(pwd, '../src'))

# And ensure modules are reloaded on any change (useful when developing code on the fly, etc)
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')


# In[2]:

import numpy as np
import pandas as pd


# ## GO demonstration

# In[3]:

from heur import GO
from objfun import Zebra3


# In[4]:

def experiment_GO(M=100, n=10, maxeval=10000):
    rows = []
    for i in range(M):        
        heur = GO(Zebra3(6), maxeval=maxeval, n=n, m=n*10, t_sel1=1, t_sel2=0.1, r=0.35, co_m=5)
        row = heur.search()
        row['maxeval'] = maxeval
        row['n'] = n
        rows.append(row)
    return pd.DataFrame(rows)


# In[5]:

tab_go = experiment_GO(M=100)


# In[6]:

def mne(x):
    return np.mean(x[x<np.inf])

def rel(x):
    return np.NaN if np.size(x) == 0 else np.size(x[x<np.inf])/np.size(x)

def feo(x):
    return np.NaN if rel(x) == 0 else mne(x)/rel(x)


# In[7]:

tab_go.pivot_table(
    values=['neval'],
    index=['n'],
    aggfunc=[rel, mne, feo]
)


# # Excercises:
# 
# 1. Implement GO
# 1. Experiment with parameters:
#    * Track and visualize their change during optimization.
#    * What is their effect on FEO? 
# 1. Compare GO and FSA, or tune GO on other objective functions
