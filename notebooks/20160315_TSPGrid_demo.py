
# coding: utf-8

# In[1]:

import numpy as np


# In[2]:

a=3
b=2


# In[3]:

p=2


# In[4]:

n=a*b
n


# In[5]:

fstar = n+np.mod(n,2)*(2**(1/p)-1)
fstar


# In[6]:

grid = np.zeros((n, 2), dtype=np.int)


# In[8]:

for i in np.arange(a):
    for j in np.arange(b):
        grid[i*b+j]=np.array([i,j], dtype=np.int) # suradnice
grid


# In[9]:

dist = np.zeros((n, n))
for i in np.arange(n):
    for j in np.arange(i+1, n):
        dist[i, j] = np.linalg.norm(grid[i, :]-grid[j, :], 2)
        dist[j, i] = dist[i, j]
print(dist)


# In[10]:

aa = np.zeros(n-1)
bb = np.arange(n-2, 0-1, -1)
print(aa)
print(bb)


# In[12]:

[np.random.randint(0, i+1) for i in np.arange(n-2, 0-1, -1)]


# In[13]:

def decode(x):
    cx = np.zeros(n, dtype=np.int)  # the final tour
    ux = np.ones(n, dtype=np.int)  # used cities indices
    ux[0] = 0  # first city is used automatically
    c = np.cumsum(ux)  # cities to be included in the tour
    for k in np.arange(1, n):
        ix = x[k-1]+1  # order index of currently visited city
        cc = c[ix]  # currently visited city
        cx[k] = cc # append visited city into final tour
        c = np.delete(c, ix)  # visited city can not be included in the tour any more
    return cx


# In[14]:

cx=decode(np.array([4,3,2,1,0]))
print(cx)


# In[15]:

cx=decode(np.array([0,0,0,0,0]))
print(cx)


# In[16]:

x = np.array([1, 2, 2, 1, 0])
cx = decode(x)
print(x, '->', cx)


# In[17]:

def tour_dist(cx):
    d=0
    for i in np.arange(n):
        dx = dist[cx[i-1], cx[i]] if i>0 else dist[cx[n-1], cx[i]]
        d += dx
        print(cx[i-1], '->', cx[i], '=', dx)
    return d
print(cx)
print(tour_dist(cx))


# In[18]:

def get_neighborhood(x, d=1):
    nd = []
    for i, xi in enumerate(x):
        # x-lower
        if x[i] > aa[i]: # (!) mutation correction .. will be discussed later
            xl = x.copy()  
            xl[i] = x[i]-1 
            nd.append(xl)
        
        # x-upper
        if x[i] < bb[i]: # (!) mutation correction ..  -- // --
            xu = x.copy()   
            xu[i] = x[i]+1 if x[i] < bb[i] else bb[i] 
            nd.append(xu)
        
    return nd


# In[19]:

print(x, ':')
print(get_neighborhood(x))

