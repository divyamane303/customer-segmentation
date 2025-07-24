import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import datetime as dt

# Load the Excel file
df = pd.read_excel("C:/Users/DIVYA/Downloads/archive/online_retail_II.xlsx")

# Drop missing customer IDs
df.dropna(subset=['Customer ID'], inplace=True)

# Keep only positive Quantity and Price
df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

# Create TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['Price']

# Create snapshot date (1 day after the last invoice date)
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Group by Customer ID to create RFM table
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
}).reset_index()

# Rename RFM columns
rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']

# Scale RFM values
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Use Elbow method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Apply KMeans clustering (choose optimal k, e.g. 4)
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Show RFM stats per cluster
print(rfm.groupby('Cluster').mean().round(2))

# Optional: Visualize clusters using PCA
pca = PCA(2)
pca_data = pca.fit_transform(rfm_scaled)

plt.figure(figsize=(8,6))
plt.scatter(pca_data[:,0], pca_data[:,1], c=rfm['Cluster'], cmap='tab10')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('Customer Segments')
plt.colorbar()
plt.show()
    