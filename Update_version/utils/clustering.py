import os
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np

# Function to display clusters based on significant distances
def cluster_and_display_points(points, significant_distances, method, data_info, save_path='Data_Pasteur/16h29_10h18_exp116_rpr/Plotting_Analysis'):
    """
    Clusters points based on the median of significant distances and displays the clusters.
    
    Parameters:
        points : A 2D array of points to be clustered
        significant_distances : A list of significant distances
        data_info : additional information to specify the data (e.g., 'data1', 'data2', etc.)
        save_path : path where the figure should be saved
    """
    
    # Case where there is no significant distances given
    if not significant_distances:
        print("No significant distances found. Clustering will not be performed.")
        return
    
    # Ensure the save directory exists
    full_save_path = os.path.join(save_path, data_info)
    os.makedirs(full_save_path, exist_ok=True)

    # Calculate the max of significant distances
    if method == 0 and len( significant_distances) > 1:
        distance = max(significant_distances)
        print("Max of significant distances from G-function:", distance)
        
    elif method == 1 and len(significant_distances) > 1:
        distance = min(significant_distances)
        print("Min of significant distances from K-function:", distance)
    
    elif method == 2 and len(significant_distances) > 1:
        distance = min(significant_distances)
        print("Min of significant distances from weighted K-function:", distance)
        
    else:
        print("No significant distances found. Clustering will not be performed.")
        return
    
    # Perform DBSCAN clustering based on the max distance
    clustering = DBSCAN(eps=distance, min_samples=2).fit(points)
    labels = clustering.labels_
    
    # Print the labels of each point
    print("Labels of each point:", labels)
    
    # Plot the clusters
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(8, 6))
    for k, col in zip(unique_labels, colors):
        class_member_mask = (labels == k)
        xy = points[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
    
    plt.title("Clusters based on significant distances")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    
    # Save the figure
    file_path = os.path.join(full_save_path, 'DBSCAN_cluster.png')
    plt.savefig(file_path)
    plt.show()