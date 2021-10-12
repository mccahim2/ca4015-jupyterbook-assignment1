#!/usr/bin/env python
# coding: utf-8

# # K-means Variation

# In[1]:


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Read in data

# In[2]:


df = pd.read_csv('data/normalise.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
df.head()


# In[3]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, labels_)


# In[163]:


pca = PCA(2)
df_tester = pca.fit_transform(df)
df_tester = pd.DataFrame(df_tester)

# Randomise order of data
df_tester = df_tester.sample(frac=1)


# In[174]:


X = df_tester[0]
Y = df_tester[1]


# In[175]:


from sklearn.model_selection import train_test_split

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[178]:


df_tester.data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




