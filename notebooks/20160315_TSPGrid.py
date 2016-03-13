
# coding: utf-8

# # Grid TSP objective function
# 
# * Cities placed on rectangular grid in $\mathbb{R}^n$, dimension given by $A, B \in \mathbb{N}$
# * (Assuming Euclidean distance) the optimal tour has length  
#   * $A \cdot B + \sqrt{2} - 1$ if $A$ and $B$ are even numbers
#   * $A \cdot B$ otherwise
# * **How to find optimal tour using heuristics?**
# 

# ## Simple example

# <img src="20160315_tspgrid.png">

# ### Our success depends mostly on efficient solution encoding (!)
# 
# * (Trivial) binary representation ... $2^{n^2}$
# * Our encoding ... $(n-1)!$ ("only")
# 
# Please **note**, that for serious TSP optimization you should use much more sophisticated approaches, e.g. *Concorde* algorithm

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
from objfun import TSPGrid


# ### Grid TSP demonstration

# ** ``TSPGrid(3, 2)`` initialisation **

# In[4]:

tsp = TSPGrid(3, 2)


# In[5]:

x = tsp.generate_point()
x


# In[6]:

cx = tsp.decode(x)
cx


# In[7]:

tsp.evaluate(x)


# In[8]:

tsp.get_neighborhood(x, 1)


# In[9]:

# optimum
tsp.evaluate(np.array([1, 2, 2, 1, 0]))


# ### TSP optimization using Random Shooting ($\mathrm{SG}_{0}$)

# In[10]:

heur = ShootAndGo(tsp, maxeval=1000, hmax=0)
print(heur.search())


# # Excercises:
# 
# 1. Implement Grid TSP on your own
# 1. Test on different dimensions
# 1. Implement your own objective function
