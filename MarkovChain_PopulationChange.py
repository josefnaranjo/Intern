#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Jose F Naranjo
# California State University - Stanislaus
# Population Change: Sacramento
# Markov Chain

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as pl


# In[2]:


# From were do the people of Sacramento come from?
# Dictionary. 

pop_sac = {'Sacramento': [0.7, 0.8, 0.6],
              'Suburbs': [0.2, 0.1, 0.4],
             'Commuter': [0.1, 0.1, 0.0]}

pop = pd.DataFrame(data = pop_sac, index = ['Sacramento', 'Suburbs', 'Commuter'])


# In[3]:


# Markov Chain:
# Sacramento to Sacramento: 70%, Sacramento to Suburbs: 20%, Commuting to another city: 10%
# Suburbs to Sacramento: 80%, Staying in the suburbs: 10%, Commuting to a different city: 10%
# Commuting from another city to Sacramento: 60%, Commuting from another city to the suburbs of Sacramento: 40%
    
movement_sim = []
movement_sim.append(pop.iloc[0].index[0])
type = np.random.choice(pop.iloc[0].index, p = pop.iloc[0])
movement_sim.append(type)

while len(movement_sim) < 70:
    type = np.random.choice(pop.iloc[pop.index.get_loc(type)].index, p = pop.iloc[pop.index.get_loc(type)])
    movement_sim.append(type)


# In[4]:


pop


# In[5]:


# Bring data to numpy array.
# How would data look like two 'pop' from now?

np.dot(pop.to_numpy(), pop.to_numpy())


# In[6]:


# Instead of doing the calculations 'by hand', use following method:

def matrix_power(matrix, power):
    if power == 0:
        return np.identity(len(matrix))
    elif power == 1:
        return matrix
    else:
        return np.dot(matrix, matrix_power(matrix, power-1))


# In[7]:


# How would data look like three 'pop' from now?

matrix_power(pop.to_numpy(), 3)


# In[8]:


# You can see that the population from Sacramento has a 71.1% chance of staying in the city. 
# The people living in the suburbs have a 19.8% chance of driving to Sacramento.
# Only 9.1% would commute to Sacramento from other cities outside the suburbs/city limits.


# In[9]:


# What would happen overtime? 
for i in range(1, 11, 1):
    print(f'\n Change in population after {i} years:\n', matrix_power(pop.to_numpy(), i), '\n')


# In[10]:


# Interestingly, overtime the population of Sacramento tends to stay in the city. The chances of moving to the suburbs diminish.
# Commuters decrease goig to the suburbs, and drive to the city instead.


# In[ ]:




