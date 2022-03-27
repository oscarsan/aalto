import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from sklearn.cluster import KMeans
#pdb.set_trace()
import pdb 
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal # Multivariate normal random variable
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Read in data from the csv file and store it in the data matrix X.
df = pd.read_csv("./data.csv")
X = df.to_numpy()

# Display first 5 rows
#print("First five datapoints:")
#display(df.head(5))


def plotting(data, centroids=None, clusters=None, title='Data', show=True):
    # This function will later on be used for plotting the clusters and centroids. But now we use it to just make a scatter plot of the data
    # Input: the data as an array, cluster means (centroids), cluster assignemnts in {0,1,...,k-1}   
    # Output: a scatter plot of the data in the clusters with cluster means
    plt.figure(figsize=(8,6))
    data_colors = ['orangered', 'dodgerblue', 'springgreen']
    centroid_colors = ['red', 'darkblue', 'limegreen'] # Colors for the centroids
    plt.style.use('ggplot')
    plt.title(title)
    plt.xlabel("feature $x_1$: customers' age")
    plt.ylabel("feature $x_2$: money spent during visit")

    alp = 0.5             # data points alpha
    dt_sz = 20            # marker size for data points 
    cent_sz = 130         # centroid sz 
    
    if centroids is None and clusters is None:
        plt.scatter(data[:,0], data[:,1], s=dt_sz, alpha=alp, c=data_colors[0])
    if centroids is not None and clusters is None:
        plt.scatter(data[:,0], data[:,1], s=dt_sz, alpha=alp, c=data_colors[0])
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=cent_sz, c=centroid_colors[:len(centroids)])
    if centroids is not None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i] for i in clusters], s=dt_sz, alpha=alp)
        plt.scatter(centroids[:,0], centroids[:,1], marker="x", c=centroid_colors[:len(centroids)], s=cent_sz)
    if centroids is None and clusters is not None:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i] for i in clusters], s=dt_sz, alpha=alp)
    
    if show:
        plt.show()

# Plot the (unclustered) data
#plotting(X) 


# m, n = X.shape    # Get the number of data points m and number of features n

# k = 3    # Define number of clusters to use

# k_means = KMeans(n_clusters = k, max_iter = 100)  # Create k-means object with k=3 clusters and using maximum 100 iterations
# k_means.fit(X)    # Fit the k-means object (find the cluster labels for the datapoints in X)
# cluster_means = k_means.cluster_centers_    # Get cluster means (centers)
# cluster_indices = k_means.labels_           # Get the cluster labels for each data point

# # Plot the clustered data
# plotting(X, cluster_means, cluster_indices)



# m, n = X.shape    # Get the number of data points m and number of features n

# k = 2    # The number of clusters to use

# np.random.seed(1)    # Set random seed for reproducability (DO NOT CHANGE THIS!)

# ### STUDENT TASK ###
# k_means = KMeans(n_clusters = k, max_iter = 100)  # Create k-means object with k=3 clusters and using maximum 100 iterations
# k_means.fit(X)    # Fit the k-means object (find the cluster labels for the datapoints in X)
# cluster_means = k_means.cluster_centers_    # Get cluster means (centers)
# cluster_indices = k_means.labels_           # Get the cluster labels for each data point

# pdb.set_trace()


# # Plot the clustered data
# plotting(X, cluster_means, cluster_indices, title='K-means clustering with $k=2$')
# print("The final cluster mean values are:\n", cluster_means)


# m = X.shape[0]  # Number of data points

# min_ind = 0  # Store here the index of the repetition yielding smallest clustering error 
# max_ind = 0  # .... largest clustering error

# cluster_assignment = np.zeros((50, m), dtype=np.int32)  # Array for storing clustering assignments
# cluster_means = []
# clustering_err = np.zeros(50,)    # Array for storing the clustering errors for each assignment

# np.random.seed(42)   # Set random seed for reproducibility (DO NOT CHANGE THIS!)

# init_means_cluster1 = np.random.randn(50,2)  # Use the rows of this numpy array to init k-means 
# init_means_cluster2 = np.random.randn(50,2)  # Use the rows of this numpy array to init k-means 
# init_means_cluster3 = np.random.randn(50,2)  # Use the rows of this numpy array to init k-means 

# best_assignment = np.zeros(m)     # Store here the cluster assignment achieving the smallest clustering error
# worst_assignment = np.zeros(m)    # Store here the cluster assignment achieving the largest clustering error

# ### STUDENT TASK ###
# # loop 0,...,49
# for i in range(50):
#   init_cluster =  np.array([init_means_cluster1[i,:], init_means_cluster2[i,:], init_means_cluster3[i,:]])
#   k_means = KMeans(n_clusters = 3, init= init_cluster, max_iter = 10, n_init=1)  # Create k-means object with k=3 clusters and using maximum 100 iterations
#   k_means.fit(X)    # Fit the k-means object (find the cluster labels for the datapoints in X)
#   cluster_means.append(k_means.cluster_centers_)    # Get cluster means (centers)
#   cluster_assignment[i] = k_means.labels_           # Get the cluster labels for each data point
#   clustering_err[i] = k_means.inertia_
  


# max_ind = np.where(clustering_err == np.amax(clustering_err))
# min_ind = np.where(clustering_err == np.amin(clustering_err))

# best_assignment = cluster_assignment[min_ind[0][0]]
# worst_assignment = cluster_assignment[max_ind[0][0]]

# best_mean = cluster_means[min_ind[0][0]]
# worst_mean = cluster_means[max_ind[0][0]]

# # Print the best and worst clustering errors
# print(f"Best clustering error: {clustering_err[min_ind]}\n")
# print(f"Worst clustering error: {clustering_err[max_ind]}\n")


# plotting(X, clusters=best_assignment, title='Cluster assignment with smallest error')
# print("Cluster assignment with largest clustering error:")
# plotting(X, clusters=worst_assignment, title='Cluster assignment with largest error')


# m = X.shape[0]    # Number of data points
# err_clustering = np.zeros(8)    # Array for storing clustering errors

# np.random.seed(1)  # Set random seed

# ### STUDENT TASK ###
# for i in range(8):
#   k_means = KMeans(n_clusters = i+1, max_iter = 100)  # Create k-means object with k=3 clusters and using maximum 100 iterations
#   k_means.fit(X)    # Fit the k-means object (find the cluster labels for the datapoints in X)
#   err_clustering[i] = k_means.inertia_
  

# print(f'Clustering errors: \n{err_clustering}')

# # Plot the clustering error as a function of the number k of clusters
# plt.figure(figsize=(8,6))
# plt.plot(range(1,9), err_clustering)
# plt.xlabel('Number of clusters')
# plt.ylabel('Clustering error')
# plt.title("The number of clusters vs clustering error")
# plt.show()    

def plot_GMM(data, means, covariances, k, clusters=None):
    
    ## Select three colors for the plot
    # if you want to plot curves k>3, extend these lists of colors
    data_colors = ['orangered', 'dodgerblue', 'springgreen'] # colors for data points
    centroid_colors = ['red', 'darkblue', 'limegreen'] # colors for the centroids
    
    k = means.shape[0]
    plt.figure(figsize=(8,6))    # Set figure size
    if clusters is None:
        plt.scatter(data[:,0], data[:,1], s=13, alpha=0.5)
    else:
        plt.scatter(data[:,0], data[:,1], c=[data_colors[i] for i in clusters], s=13, alpha=0.5)

    # Visualization of results
    x_plot = np.linspace(19, 35, 100)
    y_plot = np.linspace(0, 12, 100)
    x_mesh, y_mesh = np.meshgrid(x_plot, y_plot)
    pos = np.empty(x_mesh.shape + (2,))
    pos[:,:,0] = x_mesh 
    pos[:,:,1] = y_mesh

    # For each cluster, plot the pdf defined by the mean and covariance
    for i in range(k):
        z = multivariate_normal.pdf(pos, mean = means[i,:], cov = covariances[i])
        plt.contour(x_mesh, y_mesh, z, 4, colors=centroid_colors[i], alpha=0.5)
        plt.scatter(means[i,0], means[i,1], marker='x', c=centroid_colors[i])

    plt.title("Soft clustering with GMM")
    plt.xlabel("feature x_1: customers' age")
    plt.ylabel("feature x_2: money spent during visit")
    plt.show()



# m, n = X.shape

# # Define the number of clusters
# k = 3

# np.random.seed(1)    # Set random seed for reproducability 

# ### STUDENT TASK ###
# gmm = GaussianMixture(n_components=3)
# gmm.fit(X)

# means = gmm.means_
# covariances = gmm.covariances_
# cluster_labels = gmm.predict(X)

# plot_GMM(X, means, covariances, k, cluster_labels)
# print("The means are:\n", means, "\n")
# print("The covariance matrices are:\n", covariances)
# import numpy as np

# from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler


# # #############################################################################
# # Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(
#     n_samples=750, centers=centers, cluster_std=0.4, random_state=0
# )

# X = StandardScaler().fit_transform(X)

# # #############################################################################
# # Compute DBSCAN
# db = DBSCAN(eps=0.3, min_samples=10).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_

# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)

# print("Estimated number of clusters: %d" % n_clusters_)
# print("Estimated number of noise points: %d" % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(labels_true, labels))
# print(
#     "Adjusted Mutual Information: %0.3f"
#     % metrics.adjusted_mutual_info_score(labels_true, labels)
# )
# print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

np.random.seed(844)    # Set random seed for reproducibility

# Create dataset with separate Gaussian clusters
clust1 = np.random.normal(5, 2, (1000,2))
clust2 = np.random.normal(15, 3, (1000,2))
clust3 = np.random.multivariate_normal([17,3], [[1,0],[0,1]], 1000)
clust4 = np.random.multivariate_normal([2,16], [[1,0],[0,1]], 1000)
dataset1 = np.concatenate((clust1, clust2, clust3, clust4))

# Create dataset containing circular data
dataset2 = datasets.make_circles(n_samples=1000, factor=.5, noise=.05)[0]

# Function for plotting clustering output on two datasets
def cluster_plots(data_1, data_2, clusters_1, clusters_2, title1='Dataset 1',  title2='Dataset 2'):
    fig, ax = plt.subplots(1, 2, figsize=(12,6))
    ax[0].set_title(title1,fontsize=14)
    ax[0].set_xlim(min(data_1[:,0]), max(data_1[:,0]))
    ax[0].set_ylim(min(data_1[:,1]), max(data_1[:,1]))
    ax[0].scatter(data_1[:,0], data_1[:,1], s=13, lw=0, c=clusters_1)
    ax[1].set_title(title2,fontsize=14)
    ax[1].set_xlim(min(data_2[:,0]), max(data_2[:,0]))
    ax[1].set_ylim(min(data_2[:,1]), max(data_2[:,1]))
    ax[1].scatter(data_2[:,0], data_2[:,1], s=13, lw=0, c=clusters_2)
    fig.tight_layout()
    plt.show()

# Plot the unclustered datasets (i.e. all points belonging to cluster 1)
#cluster_plots(dataset1, dataset2, np.ones(dataset1.shape[0]), np.ones(dataset2.shape[0]))

# k_means_1 = KMeans(n_clusters=4)
# k_means_2 = KMeans(n_clusters=2)
# clusters_1 = k_means_1.fit_predict(dataset1)
# clusters_2 = k_means_2.fit_predict(dataset2)

# Plot the clustered datasets
#cluster_plots(dataset1, dataset2, clusters_1, clusters_2)

from sklearn.cluster import DBSCAN

# Define eps values for the two datasets
eps_1 = 1
eps_2 = 0.1

#Compute DBSCAN
db1 = DBSCAN(eps=eps_1, min_samples=5, metric='euclidean').fit(dataset1)
db2 = DBSCAN(eps=eps_2, min_samples=5, metric='euclidean').fit(dataset1)

clusters_1 = db1.fit_predict(dataset1)
clusters_2 = db2.fit_predict(dataset2)

labels1 = db1.labels_
labels2 = db2.labels_

dataset1_noise_points = list(labels1).count(-1)
dataset2_noise_points = list(labels2).count(-1)


print(f'Noise points in Dataset 1:\n {dataset1_noise_points}/{len(clusters_1)} \n')
print(f'Noise points in Dataset 2:\n {dataset2_noise_points}/{len(clusters_2)} \n')

# Plot the clustered datasets
cluster_plots(dataset1, dataset2, clusters_1, clusters_2)