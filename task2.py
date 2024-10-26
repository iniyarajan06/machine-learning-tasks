# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 22:04:59 2024

@author: Iniya Rajan
"""
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


df = pd.read_csv(r"C:\\Users\\Iniya Rajan\\Downloads\\archive (2)\\Mall_Customers.csv")


print(df.dtypes)
print(df.head())


df.fillna(df.mean(), inplace=True)


df = pd.get_dummies(df, columns=['Gender'], drop_first=True)


print(df.describe())

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_Male']])


k_range = range(2, 10)
sse_values = []
for k in k_range:
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)  # Set random_state for reproducibility
    kmeans.fit(df_scaled)
    sse_values.append(kmeans.inertia_)


plt.plot(k_range, sse_values)
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.title('Elbow Method for Optimal K')
plt.show()


kmeans = KMeans(n_clusters=optimal_k, n_init=10, random_state=42)
kmeans.fit(df_scaled)


labels = kmeans.labels_
silhouette_avg = silhouette_score(df_scaled, labels)
print("Silhouette score:", silhouette_avg)
