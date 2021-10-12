#!/usr/bin/env python
# coding: utf-8

# # K-Means Clustering

# ### Table of contents

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


# Read in normalised data

# In[2]:


df = pd.read_csv('data/normalise.csv')
df=df.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
df.head()


# :::{note}
# This correlation matrix is used to find highly correlated variables
# :::

# In[3]:


plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);


# ### PCA

# To avoid the curse of dimensionality. The curse of dimensionality is the problem caused by the exponential increase in volume associated with adding extra dimensions to Euclidean space. To avoid this we can perform Principal Component analysis (PCA) to our data frame. This will leave use with an array of components.
# 
# PCA is believed to imporve the performance for cluster analysis
# 
# By reducing the number of features we are increasing the performance of our algorithm
# 
# For the purpose of this assignment I will select 2 features to be used for clustering

# In[4]:


pca = PCA(2)
df_tester = pca.fit_transform(df)


# In[5]:


df_tester


# The newly optained PCA scores will be incorporated in the the K-means algorithm

# The next step of the clustering process is to determine a value for K. This can be done by:<br />
# * **The Elbow Method**
# * **Silhouette score**
# 
# These two methods will be used to find the best value for K to use for clustering

# ### Methods for finding K Value

# In[6]:


# Elbow method and silhouette method


# When the distortions are plotted and the plot looks like an arm then the [“elbow”](https://predictivehacks.com/k-means-elbow-method-code-for-python/)(the point of inflection on the curve) is the best value of k.

# The formula for the Elbow method can be seen here: <br />
# $$
# Sum Squared Errors = \sum_{i=1}^{N} {(y_i - ŷ_i)^2}
# $$

# The formula for the Silhouette Score method can be seen here: <br />
# $$
#     s_{i} = \frac{b_{i} - a_{i}}{max(b_{i}, a_{i})}
# $$

# In[7]:


wcss=[]
for i in range(1,21):
    k_means_pca=KMeans(n_clusters=i, init="k-means++", random_state=42)
    k_means_pca.fit(df)
    wcss.append(k_means_pca.inertia_)


# In[8]:


plt.figure(figsize=(10,8))
plt.plot(range(1,21), wcss, marker="o", linestyle = "--")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("K-means with PCA CLustering")
plt.show()


# In[9]:


for n in range(2, 21):
    km = KMeans(n_clusters=n)
    km.fit_predict(df)
    score = silhouette_score(df, km.labels_, metric='euclidean')
    print('N = ' + str(n) + ' Silhouette Score: %.3f' % score)


# From the various methods above the optimal value for K to us used for clustering will be 4

# ### Implementation of the K-means Clustering Process

# In[10]:


k_means_pca = KMeans(n_clusters=4, init="k-means++", random_state=42)


# In[11]:


k_means_pca.fit(df_tester)


# In[12]:


df_segm_pca_kmeans = pd.concat([df.reset_index(drop=True), pd.DataFrame(df_tester)], axis=1)
df_segm_pca_kmeans.columns.values[-2:] = ["Component 1", "Component 2"]

df_segm_pca_kmeans["Segment K-means PCA"] = k_means_pca.labels_


# In[13]:


df_segm_pca_kmeans.head()


# In[14]:


df_segm_pca_kmeans["Segment K-means PCA"].value_counts()


# In[15]:


df_segm_pca_kmeans["Segment"] = df_segm_pca_kmeans["Segment K-means PCA"].map({0:"first", 1:"second", 2:"third", 3:"fourth"})


# In[16]:


df_segm_pca_kmeans["Segment"].value_counts()


# In[17]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (10, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["Segment"], palette=["g", "r", "c", "m"])
plt.title("Clusters by PCA Components")
plt.show()


# We can now observe the seperate clusters

# With the Clusters seperated we are able to visualise almost all the entire data set

# Before showing cluster analysis, change study types back to their original names

# In[18]:


df_original = pd.read_csv('data/all_data.csv')
df_original=df_original.drop(["Unnamed: 0"], axis=1) # Drop Unnamed: 0 column
df_original.head(1)


# In[19]:


df_segm_pca_kmeans["Study_Name"] = df_original["Study_Type"]
df_segm_pca_kmeans["Study_Name"]


# In[20]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["Study_Name"], palette=["b", "c", "g", "k", "m", "r", "y", "darkmagenta", "greenyellow", "navy"])
plt.title("Clusters by Study Type")
plt.show()


# The first observation to note is that the Steingroever2011 study and Wetzels study are out on their own in the scatter plot.
# 
# They are both part of segments 3 and 4. Both of these studies contained 150 participants.

# From looking at the way the Studies have clustered it is clear to see that they all follow the same pattern

# In[21]:


df_segm_pca_kmeans["Total_won_vs_lost"] = df_original["Total"]


# In[22]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["Total_won_vs_lost"])
plt.title("Clusters by Total won/lost")
plt.show()


# From analysing this scatter plot it is clear to see that there is a reduction in total won as component 1 gets bigger

# In[23]:


df_segm_pca_kmeans["deck1"] = df_original["1"]
df_segm_pca_kmeans["deck2"] = df_original["2"]
df_segm_pca_kmeans["deck3"] = df_original["3"]
df_segm_pca_kmeans["deck4"] = df_original["4"]


# In[24]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["deck1"])
plt.title("Clusters by Study Type")
plt.show()


# In[25]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["deck2"])
plt.title("Clusters by Selecting deck 2")
plt.show()


# In[26]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["deck3"])
plt.title("Clusters by Selecting deck 3")
plt.show()


# In[27]:


x_axis = df_segm_pca_kmeans["Component 1"]
y_axis = df_segm_pca_kmeans["Component 2"]
plt.figure(figsize = (15, 8))
sns.scatterplot(x_axis, y_axis, hue = df_segm_pca_kmeans["deck4"])
plt.title("Clusters by Selecting deck 4")
plt.show()


# This is an example of a cluster made without using the Principal Component analysis.
# 
# It is evident that the groups are all mixed together, therefore making it harder to determine any proper findings from the clusters.

# In[28]:


kmeans_margin_standard = KMeans(n_clusters=4).fit(df[["Total", "Study_Type"]])
centroids_betas_standard = kmeans_margin_standard.cluster_centers_


# In[29]:


plt.figure(figsize=(16,8))
plt.scatter(df['Total'], df['2'], c= kmeans_margin_standard.labels_, cmap = "Set1", alpha=0.5)
plt.scatter(centroids_betas_standard[:, 0], centroids_betas_standard[:, 1], c='blue', marker='x')
plt.title('K-Means cluster for all Subjects - Most Common Choice Picked')
plt.xlabel('Total')
plt.ylabel('Times Deck 2 Picked')
plt.show()


# Put total values into bins for examination approx 10 bins

# ### Conclusions

# After running my **Principal component analysis** it made the clustering phase a lot easier.
# 
# It allowed me to interpret results easier
# 
# From analysing all the cluster graphs above it is clear to see that there are some findings to take away.
# 
# When taking into account, totals with the chosen decks, it is expected that decks 3 and 4 would be selected the most for the participants who made the most monet and this is evident in the cluster maps above.
