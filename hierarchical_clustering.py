from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)
linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(100, 100),dpi=100) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=list(tfidf_df.index));#labels are rows of the tfidf  matrix i.e. documents

plt.tick_params(
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()
#uncomment below to save figure
plt.savefig('ward_clusters_features.png', dpi=100)

'''
A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i. A cluster with an index less than n corresponds to one of the original observations. The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2]. The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.
'''

#Dendogram Truncation
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    linkage_matrix,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=12,  # show only the last p merged clusters
    show_leaf_counts=False,  # otherwise numbers in brackets are counts
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()
#Reference: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


#Selecting a Distance Cut-Off aka Determining the Number of Clusters
# set cut-off to 50
max_d = 50  # max_d as in max_distance
fancy_dendrogram(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10,
    max_d=max_d,  # plot a horizontal cut-off line
)
plt.show()

#Use the 
#1) Plot obtained above,
#2) Elbow Method, 
#3) Inconsistency Method to determine number of cluters
#Link: https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

#Using number of clusters to make hierarchical clusters
k=30
clusters = fcluster(linkage_matrix, k, criterion='maxclust')
Counter(clusters)

#Using Distance to make Hierarchical Clusters
max_d = 50 #maximum distance
clusters_dist = fcluster(linkage_matrix, max_d, criterion='distance')
clusters_dist
Counter(clusters_dist)

#Ploting the clusters obtained
#If have a 2D matrix X having data points of the form (x,y)
plt.figure(figsize=(10, 8))
plt.scatter(X[:,0], X[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors
plt.show()

