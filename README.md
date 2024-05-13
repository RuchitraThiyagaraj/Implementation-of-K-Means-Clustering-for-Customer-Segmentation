# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries: pandas for data manipulation and matplotlib.pyplot for visualization. 
2. Load the customer data from a CSV file into a pandas DataFrame.
3. Perform K-means clustering using sklearn's KMeans algorithm, employing the elbow method to find the optimal number of clusters.
4. Fit a KMeans model with 5 clusters to the data.
5. Predict cluster labels for each data point and add them to the DataFrame.
6. Visualize the customer segments by plotting their annual income against spending score, coloring points by cluster membership.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: RUCHITRA T
RegisterNumber: 212223110043
*/
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Mall_Customers.csv")
data
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
  Kmeans=KMeans(n_clusters = i,init = "k-means++")
  Kmeans.fit(data.iloc[:,3:])
  wcss.append(Kmeans.inertia_)
plt.plot(range(1, 11), wcss[:10])
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km=KMeans(n_clusters = 5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Score (1-100)"],c="red",label="cluster0")
plt.scatter(df1["Annual Income (k$)"],df1["Spending Score (1-100)"],c="black",label="cluster1")
plt.scatter(df2["Annual Income (k$)"],df2["Spending Score (1-100)"],c="blue",label="cluster2")
plt.scatter(df3["Annual Income (k$)"],df3["Spending Score (1-100)"],c="green",label="cluster3")
plt.scatter(df4["Annual Income (k$)"],df4["Spending Score (1-100)"],c="magenta",label="cluster4")
plt.legend()
plt.title("Customer Segments")
```

## Output:
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/957a5b45-6e2d-49b2-aeaf-076fbec81a93)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/424c6ede-ee99-45d1-95b5-43eb5a338722)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/ea2caa0c-0f95-4f87-83e8-49e2728621b5)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/62da6e8c-21fe-448e-b231-96438ed31817)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/92e58bf9-a4ce-4b0f-8ba3-a6832687d482)
![image](https://github.com/RuchitraThiyagaraj/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/154776996/979404ef-ebea-4077-8cf7-78eebdab134c)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
