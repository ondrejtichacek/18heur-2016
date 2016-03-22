
# coding: utf-8

# ### Fast Simulated Annealing
# 
# Few hints:
# 
# * $T_0 \in \mathbb{R}$ initial temperature
# * $ n_0 \in \mathbb{N} $ and $ \alpha \in \mathbb{R} $  - cooling strategy parameters
# * $k$-th step:
#   * **Mutate** the solution ``x`` -> ``y``
#   * $ T = \frac{T_0}{1+(k/n_0)^\alpha} $ for $ \alpha > 0 $,
#   * $ T = T_0 \cdot \exp(-(k/n_0)^{-\alpha}) $ otherwise.
#   * $s = \frac{f_x-f_y}{T}$
#   * replace ``x`` if $u < 1/2 + \arctan(s)/\pi$ where $u$ is random uniform number
#   
# Cauchy **mutation operator**:
# 
# * mutation perimeter (width) controlled by parameter $R \in \mathbb{R}$
# * $ \boldsymbol{\mathsf{x}}_\mathrm{new} = \boldsymbol{\mathsf{x}} + R \cdot \tan{\left(\pi \left(\boldsymbol{\mathsf{r}} - \dfrac{1}{2}\right)\right)} $ where $\boldsymbol{\mathsf{r}}$ is random uniform vector

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


# ## FSA demonstration

# In[3]:

from heur import FSA
from objfun import TSPGrid, Zebra3


# In[4]:

heur = FSA(TSPGrid(3, 2), maxeval=100, T0=0.1, n0=100, alpha=2, r=0.5)
print(heur.search())


# In[5]:

heur = FSA(Zebra3(6), maxeval=10000, T0=0.1, n0=100, alpha=2, r=0.5)
print(heur.search())


# # Excercises:
# 
# 1. Implement FSA
# 1. Experiment with parameters:
#    * Track and visualize their change during optimization.
#    * What is their effect on FEO? 
#    * How does FEO change on different objective functions?
# 1. Compare FSA with Shoot and Go
# 1. Try different mutations, e.g. Gauss one. Or Lévy Flight.
# 1. Could you come up with different mutation correction?
