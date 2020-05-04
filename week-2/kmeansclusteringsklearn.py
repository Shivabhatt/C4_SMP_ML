import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

data=pd.read_csv('ex2data3.csv')                                                                        #Reading in the data to be tested

sns.set_style('darkgrid')

plot1 = plt.figure(1)
plt.scatter(data['Satisfaction'], data['Loyalty'])                                                      #plotting out the data
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

x=data.copy()                                                                                           #copying the data so the original data so we don't actually change it
kmeans=KMeans(2)                                                                                        #Setting number of clusters to 2
cluster_pred = kmeans.fit_predict(x)                                                                    #fit and predict
plot2 = plt.figure(2)
plt.scatter(x['Satisfaction'],x['Loyalty'],c = cluster_pred , cmap='rainbow')                           #plotting out the cluster
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')


from sklearn import preprocessing                                                                       #preprocessing the data to be scaled, implemented by sklearn
x_scaled=preprocessing.scale(x)

wcss=[]

'''Implementing the elbow method to 
find the adequate number of clusters'''

for i in range(1,30):
  kmeans=KMeans(i)
  kmeans.fit(x_scaled)
  wcss.append(kmeans.inertia_)
'''Plotting out the number of
clusters vs the wcss value '''
plot3 = plt.figure(3)
plt.plot(range(1,30),wcss, marker = 'x', mec= 'red')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
#plt.show()

'''Take the x_scaled data and fit it into a cluster of size four and check the result'''
km = KMeans(n_clusters = 4)

cluster_pred = km.fit_predict(x_scaled)
plot4 = plt.figure(4)
plt.scatter(x['Satisfaction'],x['Loyalty'], c = cluster_pred, cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')

plt.show()
