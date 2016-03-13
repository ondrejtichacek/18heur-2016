
# coding: utf-8

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
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')  # for inline plots
import seaborn as sbn


# In[3]:

from heur import ShootAndGo
from objfun import Zebra3


# ### Run ``maxeval`` vs ``hmax`` experiments

# In[4]:

def experiment(M, maxevals, hmaxs):
    rows = []
    for maxeval in maxevals:
        for hmax in hmaxs:
            for i in range(M):
                heur = ShootAndGo(Zebra3(6), maxeval=maxeval, hmax=hmax)
                row = heur.search()
                row['maxeval'] = maxeval
                row['hmax'] = hmax
                rows.append(row)
    return pd.DataFrame(rows)


# In[5]:

M = 200
maxevals = np.array([1000, 10000, 50000, 100000, 200000, np.inf])
hmaxs = np.array([0, 1, 5, 10, 50, np.inf])


# In[6]:

## This way we can run experiments...
#%%time
#tab = experiment(M, maxevals, hmaxs)
#
## And save them into file for future analysis
#tab.to_csv('20160315_zebra_heatmap.csv')


# In[7]:

# But now let's load them from file directly
tab = pd.read_csv('20160315_zebra_heatmap.csv')


# In[8]:

tab.head()


# ### Analyze ``maxeval`` vs ``hmax`` experiment results

# #### 1. Reliability

# In[9]:

map_rel = tab.pivot_table(
    values=['neval'],
    index=['hmax'],
    columns=['maxeval'],
    aggfunc=lambda x: len(x[x<np.inf])/len(x)
)['neval']
map_rel


# In[10]:

sbn.heatmap(map_rel, annot=True, fmt=".2f")


# #### 2. Mean number of evaluations

# In[11]:

map_mne = tab.pivot_table(
    values=['neval'],
    index=['hmax'],
    columns=['maxeval'],
    aggfunc=lambda x: np.mean(x[x<np.inf])
)['neval']
map_mne


# In[12]:

sbn.heatmap(map_mne, annot=True, fmt=".0f")


# #### 3. Feoktistov criterion

# In[13]:

map_feo = tab.pivot_table(
    values=['neval'],
    index=['hmax'],
    columns=['maxeval'],
    aggfunc=lambda x: np.NaN if len(x[x<np.inf])/len(x) ==0 else np.mean(x[x<np.inf])/(len(x[x<np.inf])/len(x))
)['neval']
map_feo


# In[14]:

sbn.heatmap(map_feo, annot=True, fmt=".0f")

