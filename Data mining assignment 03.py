#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt



file_path = input("Enter the path of the dataset file: ")

if file_path=='./UCI_dataset/yeast_training.txt':

    # Provide column names 
    column_names = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]  # Replace these with your actual column names

    # Read the text file into a DataFrame
    df = pd.read_csv(file_path,header=None, names=column_names, delim_whitespace=True)


elif file_path=='./UCI_dataset/pendigits_training.txt':
    # Provide column names
    column_names = columns = ["feature_{}".format(i) for i in range(16)] + ["class"]

    # Read the text file into a DataFrame
    df = pd.read_csv(file_path,header=None, names=column_names, delim_whitespace=True)


    
else:
    file_path=='./UCI_dataset/satellite_training.txt'
    # Provide column names
    column_names = columns = ["feature_{}".format(i) for i in range(36)] + ["class"]

    # Read the text file into a DataFrame
    df = pd.read_csv(file_path,header=None, names=column_names, delim_whitespace=True)

# Convert the class column to numeric
df['class'] = pd.Categorical(df['class']).codes 


# Step 3: Implement the K-means algorithm without using a library function
def k_means(data, k, max_iterations=20, random_state=0):
    np.random.seed(random_state)
    
    # Randomly initialize centroids
    centroids = data.sample(n=k, random_state=random_state)
    
    for _ in range(max_iterations):
        # Calculate Euclidean distances
        distances = np.linalg.norm(data.values[:, np.newaxis] - centroids.values, axis=2)
        
        # Assign each point to the closest centroid
        labels = np.argmin(distances, axis=1)
        
        # Update centroids
        centroids = data.groupby(labels).mean()
    
    return labels, centroids

# Collect SSE values for different values of K
k_values = list(range(2, 11))
sse_values = []

#calvulate SSE using euclidiean distance
for k in range(2, 11):
    labels, centroids = k_means(df.drop(columns=["class"]), k)
    sse = np.sum(np.linalg.norm(df.drop(columns=["class"]).values - centroids.values[labels], axis=1))
    sse_values.append(sse)
    print(f"For K={k}, After 20 iterations: SSE error={sse:0.4f}")

# Plot the Error vs k chart
plt.plot(k_values, sse_values, marker='o')
plt.title('Error vs K')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('SSE Error')
plt.show()


