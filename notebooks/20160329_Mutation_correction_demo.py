
# coding: utf-8

# # Mutation correction

# Let's imagine that we want to mutate vector $(4, 5)$ with mask $(1, -3)$. That would result in vector $(5, 2)$ which lays out of domain between $\boldsymbol{\mathsf{a}} = (3, 3)$ and $ \boldsymbol{\mathsf{b}} = (9, 8)$.
# 
# Following image illustrates this exact situation:

# <img src="20160329_correction-example.gif">

# We have three straightforward options:
# 
# 1. cut off the new vector on the edge of the domain, i.e. $\boldsymbol{\mathsf{x_\mathrm{new}}} = (5,3)$,
# 2. read the new vector from periodic expansion of the domain, i.e. $\boldsymbol{\mathsf{x_\mathrm{new}}} = (5,7)$,
# 3. read the new vector as mirror image of the old vector back into the domain i.e. $\boldsymbol{\mathsf{x_\mathrm{new}}} = (5,4)$.
# 
# Graphically:

# <img src="20160329_correction-strategies.gif">

# Currently we have implemented only the first option, **let's now implement the other two**.
