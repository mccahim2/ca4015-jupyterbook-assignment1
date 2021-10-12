#!/usr/bin/env python
# coding: utf-8

# # Data preparation for clustering

# For clustering using the k-means method we will need to follow a few steps:
# * **Standardise the data:** The process of converting an actual range of values into a standard range of values
# * **Find a Similarity Measure:**
# * **Interpret Results:**

# In[1]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# Importing cleaned data

# In[2]:


all_data = pd.read_csv('data/all_data.csv')
all_data=all_data.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
all_data.head()

df = pd.read_csv('data/all_data.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column


# Changing categorical data entries can be done by One-hot encoding.
# 
# **Label encoding** is a method during data preparation for converting categorical data variables so they can be provided to machine learning algorithims to improve predictions
# 
# LabelEncoder() is a data manipulation function used to convert categorical data into indicator variables
# 
# :::{note}
# Machine learning models require all input and output variables to be numeric
# :::
# 
# It is impossible to do k-means clustering on a categorical variable

# In[3]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
df["Study_Type"] = labelencoder.fit_transform(df["Study_Type"])
df.head()


# In[4]:


df["Study_Type"].value_counts()


# I opted for LabelEncoder as opposed to One-Hot Encoder to reduce the number of demensions being used in the data set.
# 
# This will be important for clustering due to the fact that k-means clustering can suffer from the curse of dimensionality

# In[5]:


all_data.head()


# ### Standardising data

# As for K-means, often it is not sufficient to normalize only mean. One normalizes data equalizing variance along different features as K-means is sensitive to variance in data, and features with larger variance have more emphasis on result. So for K-means, I would recommend using StandardScaler for data preprocessing.
# 
# Don't forget also that k-means results are sensitive to the order of observations, and it is worth to run algorithm several times, shuffling data in between, averaging resulting clusters and running final evaluations with those averaged clusters centers as starting points.

# ### Standardising data Test

# In[6]:


scaler = preprocessing.StandardScaler()
segmentation_std = scaler.fit_transform(df)


# In[7]:


segmentation_std


# The standardised data is now stored in an array. I will convert it back to a pandas dataframe,

# In[8]:


df_standard = pd.DataFrame(segmentation_std, columns=['Total', 'Study_Type', 'No_participants', 'Amount_won', 'Amount_lost', '1', '2', '3', '4'])
df_standard


# **Exporting Data to CSV file**

# In[9]:


df_standard.to_csv('data/normalise.csv')

