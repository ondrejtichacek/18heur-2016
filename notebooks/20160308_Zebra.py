
# coding: utf-8

# # Clerc's Zebra-3 objective function
# 
# Clerc's Zebra-3 problem is a non-trivial binary optimization problem and part of discrete optimization benchmark problems (Hierarchical swarm model: a new approach to optimization, Chen et al, 2010).
# 
# Zebra-3 function is defined for $d = 3 \, d^*$, $d^* \in \mathbb{N}$ as
# $$ \mathrm{z}(\boldsymbol{\mathsf{x}}) = \sum_{k=1}^{d^*} \mathrm{z}_{1+\mathrm{mod}(k-1,2)} (\boldsymbol{\mathsf{\xi}}_k) $$
# where
# $\boldsymbol{\mathsf{\xi}}_k = (x_{3\,k-2}, \ldots, x_{3\,k})$ and
# 
# $$
# \mathrm{z_1}(\boldsymbol{\mathsf{\xi}}) = \left\{
# \begin{array}{c l}     
#     0.9 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=0 \\
#     0.6 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=1 \\
#     0.3 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=2 \\
#     1.0 & \  \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | | _1=3  
# \end{array}\right.
# $$
# 
# $$
# \mathrm{z_2}(\boldsymbol{\mathsf{\xi}}) = \left\{
# \begin{array}{c l}     
#     0.9 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=3 \\
#     0.6 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=2 \\
#     0.3 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=1 \\
#     1.0 & \mathrm{for} \  | | \boldsymbol{\mathsf{\xi}} | |_1=0 
# \end{array}\right.
# $$
# 
# Zebra-3 function is a subject of maximization with maximum value of $d/3$. Therefore  we will minimize 
# 
# $$\mathrm{f}(\boldsymbol{\mathsf{x}})=\frac{d}{3} - \mathrm{z}(\boldsymbol{\mathsf{x}})$$
# 
# with $f^* = 0$.

# <img src="20160308_zebra3.png">

# ### Set up IPython notebook environment first...

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


# In[3]:

from heur import ShootAndGo
from objfun import Zebra3


# ### Zebra-3 demonstration

# ** ``Zebra3(6)`` initialisation **

# In[4]:

z3 = Zebra3(6)


# I.e. we have $2^{6 \cdot 3} = 262 144$ states to search

# In[5]:

x = z3.generate_point()
x


# In[6]:

z3.evaluate(x)


# In[7]:

z3.get_neighborhood(x, 1)


# In[8]:

# optimum
z3.evaluate(np.array([0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1]))


# ### Zebra-3 optimization using $\mathrm{SG}_{100}$

# In[9]:

heur = ShootAndGo(Zebra3(6), maxeval=10000, hmax=100)
print(heur.search())


# # Excercises:
# 
# 1. Find the best ``hmax`` for ``maxeval=10000``
# 1. Does choice of ``maxeval`` affect optimal ``hmax``?
# 1. Implement your own objective function
