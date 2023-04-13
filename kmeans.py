# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 14:54:47 2023

@author: JESSICA
"""

#importing 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#importingdata
data=pd.read_csv('placement.csv')
X=data.values


#visualizingdatapoints
plt.figure(figsize=(15,8))
sns.scatterplot(X[:,0], X[:, 1])
plt.xlabel('cgpa')
plt.ylabel('placements')
plt.show()

#wcss
wcss=[]
for i in range(1,10):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=2)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
    
#plotting to find k value
plt.figure(figsize=(15,8))
plt.plot(range(1,10),wcss)

#TRAINING
kmeans=KMeans(n_clusters=3,init="k-means++",random_state=1)
Y=kmeans.fit_predict(X)

#Centroid points
kmeans.cluster_centers_

#plotting the predcited values
plt.figure(figsize=(15,8))
plt.scatter(X[Y==0,0],X[Y==0,1],c='red',label='cluster1')
plt.scatter(X[Y==1,0],X[Y==1,1],c='green',label='cluster2')
plt.scatter(X[Y==2,0],X[Y==2,1],c='pink',label='cluster3')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],c='black',s=200)

#newdata point
#new=[[7.88,55]]
#A=kmeans.predict(new)
#plt.scatter(A,[1],s=300,c='blue')






    
