#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Assignment by Chidimma Idika

# In[3]:


import numpy as np
import pandas as pd


# ## Question 1
# What's the version of Pandas that you installed?

# In[5]:


print(pd.__version__)


# ### Getting the data
# - For this homework, we'll use the California Housing Prices dataset. 
# - wget https://raw.githubusercontent.com/alexeygrigorev/datasets/master/housing.csv

# In[6]:


df = pd.read_csv('raw.githubusercontent.com_alexeygrigorev_datasets_master_housing.csv')
df.head()


# ## Question 2
# How many columns are in the dataset?
# 
# - Ans: 10
# 

# In[12]:


df.columns


# ## Question 3
# Which columns in the dataset have missing values?
# 
# - total_rooms
# - total_bedrooms
# - both of the above
# - no empty columns in the dataset

# In[15]:


df.head(2)


# In[17]:


df.isnull().sum()


# - Ans: total_bedrooms

# ## Question 4
# How many unique values does the ocean_proximity column have?
# 
# - 3
# - 5
# - 7
# - 9

# In[18]:


df.ocean_proximity.nunique()


# ## Question 5
# What's the average value of the median_house_value for the houses located near the bay?
# 
# - 49433
# - 124805
# - 259212
# - 380440

# In[22]:


bay_area_houses = df[df['ocean_proximity'] == 'NEAR BAY']
bay_area_houses['median_house_value'].mean().round()


# ## Question 6
# 1. Calculate the average of total_bedrooms column in the dataset.
# 2. Use the fillna method to fill the missing values in total_bedrooms with the mean value from the previous step.
# 3. Now, calculate the average of total_bedrooms again.
# 4. Has it changed?
# 
# Has it changed?
# 
# ```Hint: take into account only 3 digits after the decimal point.```
# 
# - Yes
# - No

# In[23]:


df.head(2)


# In[28]:


avg1 = df['total_bedrooms'].mean().round(3)
avg1


# In[29]:


df['total_bedrooms'].fillna(avg1, inplace=True)


# In[30]:


avg2 = df['total_bedrooms'].mean().round(3)
avg2


# No, the average did not change

# ## Question 7
# 1. Select all the options located on islands.
# 2. Select only columns housing_median_age, total_rooms, total_bedrooms.
# 3. Get the underlying NumPy array. Let's call it X.
# 4. Compute matrix-matrix multiplication between the transpose of X and X. To get the transpose, use X.T. Let's call the result XTX.
# 5. Compute the inverse of XTX.
# 6. Create an array y with values [950, 1300, 800, 1000, 1300].
# 7. Multiply the inverse of XTX with the transpose of X, and then multiply the result by y. Call the result w.
# 8. What's the value of the last element of w?
# 
# 
# - -1.4812
# - 0.001
# - 5.6992
# - 23.1233

# In[49]:


island_df = df[df['ocean_proximity'] == 'ISLAND']
island_df


# In[51]:


select_df = island_df[['housing_median_age', 'total_rooms', 'total_bedrooms']]
select_df


# In[53]:


X = np.array(select_df)
X


# In[63]:


X_transpose = X.T
X_transpose

XTX = X_transpose.dot(X)
XTX


# In[67]:


XTX_inv = np.linalg.inv(XTX)
XTX_inv


# In[69]:


y = np.array([950, 1300, 800, 1000, 1300])
y


# In[72]:


w = (XTX_inv.dot(X_transpose)).dot(y)
w


# The value of the last element of w = 5.6992
