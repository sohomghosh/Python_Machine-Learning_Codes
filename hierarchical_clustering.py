from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(tfidf_matrix)

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(100, 100),dpi=100) # set size
ax = dendrogram(linkage_matrix, orientation="right", labels=list(features));

plt.tick_params(
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()
#uncomment below to save figure
plt.savefig('ward_clusters_roles.png', dpi=100)

