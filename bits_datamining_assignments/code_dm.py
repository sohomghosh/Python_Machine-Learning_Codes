#NOTE : data folder contains all csv files mentioned in the assignment

import os
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import tree, metrics
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import ward, dendrogram
import matplotlib as mpl
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import fcluster
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from sklearn.linear_model import LinearRegression


path = '/Users/sohom.ghosh/Documents/HS_M/0_BITS_PILANI/Sem-2/DataMining/Assignment/data/'

df = pd.read_csv(path+'prop-6.csv',header=0)
df.columns = [str(cl).lower() for cl in df.columns]
indexes = {"prop-6.csv" : [0, df.shape[0]]}
ind_cnt = df.shape[0]

li = os.listdir(path)
li.remove('prop-6.csv')


#reading data from all csv
for file in li:
    data = pd.read_csv(path+file,header=0)
    df = df.append(data)
    indexes[file] = [ind_cnt, ind_cnt + data.shape[0]]
    ind_cnt = ind_cnt + data.shape[0]


df.columns

#name_combined with version may be a good determining factor for predicting bugs
df['name_version'] = df['name'].map(str) + df['version'].map(str)
df['full_name'] = df['name.1'].apply(lambda x: str(x).lower().strip())

#one hot encoding for features like name_version and full_name
one_hot_features = pd.get_dummies(df[['name_version','full_name']])
x = df.drop(['bug'], axis=1)
x = x._get_numeric_data()
x = pd.concat([x, one_hot_features], axis=1)
y = df.bug

pd.scatter_matrix(x[['amc', 'avg_cc', 'ca', 'cam', 'cbm', 'cbo', 'ce', 'dam', 'dit', 'ic', 'lcom', 'lcom3', 'loc', 'max_cc', 'mfa', 'moa', 'noc', 'npm', 'rfc', 'wmc']])
plt.show()

plt.clf()

for i in ['amc', 'avg_cc', 'ca', 'cam', 'cbm', 'cbo', 'ce', 'dam', 'dit', 'ic', 'lcom', 'lcom3', 'loc', 'max_cc', 'mfa', 'moa', 'noc', 'npm', 'rfc', 'wmc']:
	sns.boxplot(data=x, x = i)
	plt.savefig("/Users/sohom.ghosh/Documents/HS_M/0_BITS_PILANI/Sem-2/DataMining/Assignment/box_plot_of_"+i+".png")


###Future works : remove outliers treatment based on this box plots
'''
for i in ['amc', 'avg_cc', 'ca', 'cam', 'cbm', 'cbo', 'ce', 'dam', 'dit', 'ic', 'lcom', 'lcom3', 'loc', 'max_cc', 'mfa', 'moa', 'noc', 'npm', 'rfc', 'wmc']:
    q1 = float(x[i].quantile([0.25]))
    q3 = float(x[i].quantile([0.75]))
    x[i] = x[i].apply(lambda x : q3 if float(x) > q3 else q1 if float(x) < q1 else float(x))
    '''

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)
X_full = x
Y_full = y

#########################1) Decision Tree
clf = tree.DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False)

clf = clf.fit(X_train, y_train)

train_dt_pred = clf.predict(X_train)
train_dt_pred = [round(i) for i in train_dt_pred]

test_dt_pred = clf.predict(X_test)
test_dt_pred = [round(i) for i in test_dt_pred]

full_test_dt_pred = clf.predict(X_full)
full_test_dt_pred = [round(i) for i in full_test_dt_pred]

#validation
print(metrics.mean_squared_error(y_test,test_dt_pred))
#8.406164383561643

##########################2) K-Means Clustering
scaler = StandardScaler()
x_km = scaler.fit_transform(X_train)
x_km_test = scaler.transform(X_test)
x_km_full = scaler.transform(X_full)
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(x_km)
    distortions.append(sum(np.min(cdist(x_km, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / x_km.shape[0])


# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

#Conclsuion from the figure elbow_kmeans.png i.e. optimal number of clusters = 7
kmeans = KMeans(n_clusters = 7)
kmeans.fit(x_km)
train_kmeans_pred = list(kmeans.predict(x_km))
test_kmeans_pred = list(kmeans.predict(x_km_test))
full_test_kmeans_pred = list(kmeans.predict(x_km_full))


##########################3) Hierarchical Clustering
hc_features = ['amc', 'avg_cc', 'ca', 'cam', 'cbm', 'cbo', 'ce', 'dam', 'dit', 'ic', 'lcom', 'lcom3', 'loc', 'max_cc', 'mfa', 'moa', 'noc', 'npm', 'rfc', 'wmc']
hc_input_train = X_train[hc_features]
hc_input_test = X_test[hc_features]
hc_input_full = X_full[hc_features]

dist_train = 1 - cosine_similarity(hc_input_train)
dist_test = 1 -  cosine_similarity(hc_input_test)
dist_full = 1 - cosine_similarity(hc_input_full)

linkage_matrix_train = ward(dist_train) #define the linkage_matrix using ward clustering pre-computed distances
linkage_matrix_test = ward(dist_test)
linkage_matrix_full = ward(dist_full)

fig, ax = plt.subplots(figsize=(100, 100),dpi=100) # set size
ax = dendrogram(linkage_matrix_train, orientation="right");

plt.tick_params(
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()

k=8 # from the plot done above
train_hc_clusters = list(fcluster(linkage_matrix_train, k, criterion='maxclust'))
test_hc_clusters = list(fcluster(linkage_matrix_test, k, criterion='maxclust'))
full_hc_clusters = list(fcluster(linkage_matrix_full, k, criterion='maxclust'))

##########################4) Fuzzy c-means clsutering
mdl = FuzzyKMeans(k=7, m=2)
mdl.fit(np.concatenate([x_km,x_km_test],axis = 0))
train_fcm_pred = list(mdl.labels_)[0:x_km.shape[0]]
test_fcm_pred = list(mdl.labels_)[x_km.shape[0]:]
mdl.fit(x_km_full)
full_fmc_pred = list(mdl.labels_)


##########################Stacking outputs of Decision Tree, K-means clustering, hierarchical clustering, fuzzy c-means clustering for the final prediction################
x_train_features = pd.DataFrame({'decision_tree_output' : train_dt_pred, 'kmeans_output' : train_kmeans_pred, 'hc_output' : train_hc_clusters, 'fcm_output' : train_fcm_pred})
x_test_features =pd.DataFrame({'decision_tree_output' : test_dt_pred, 'kmeans_output' : test_kmeans_pred, 'hc_output' : test_hc_clusters, 'fcm_output' : test_fcm_pred})
y_train 
y_test

linreg=LinearRegression()
linreg.fit(x_train_features,y_train)
y_predict_final = linreg.predict(x_test_features)
print(metrics.mean_squared_error(y_test,y_predict_final))
#Ans: 3.03


#conclusion: mean squared error reduced from 8.4 to 3.09



def for_individual_files(file_name):
    indexes_file = indexes[file_name]
    test_dt_file = full_test_dt_pred[indexes_file[0] : indexes_file[1]]
    test_kmeans_file = full_test_kmeans_pred[indexes_file[0] : indexes_file[1]]
    test_hc_file = full_hc_clusters[indexes_file[0] : indexes_file[1]]
    test_fuz_file = full_fmc_pred[indexes_file[0] : indexes_file[1]]
    x_test_file = pd.DataFrame({'decision_tree_output' : test_dt_file, 'kmeans_output' : test_kmeans_file, 'hc_output' : test_hc_file, 'fcm_output' : test_fuz_file})
    y_predict_file = [1 if i>=1 else 0 for i in linreg.predict(x_test_file)]
    y_full_file = Y_full[indexes_file[0] : indexes_file[1]]
    y_full_file = [1 if i>=1 else 0 for i in y_full_file]
    return (metrics.accuracy_score(y_full_file,y_predict_file), metrics.f1_score(y_full_file,y_predict_file))




for i in ['prop-6.csv', 'arc.csv', 'berek.csv', 'camel-1.6.csv', 'e-learning.csv', 'forrest-0.8.csv', 'intercafe.csv', 'ivy-2.0.csv', 'jedit-4.3.csv', 'kalkulator.csv', 'log4j-1.2.csv', 'lucene-2.4.csv', 'nieruchomosci.csv', 'pbeans2.csv', 'pdftranslator.csv', 'poi-3.0.csv', 'redaktor.csv', 'serapion.csv', 'skarbonka.csv', 'synapse-1.2.csv', 'systemdata.csv', 'szybkafucha.csv', 'termoproject.csv', 'tomcat.csv', 'velocity-1.6.csv', 'workflow.csv', 'xalan-2.7.csv', 'xerces-1.4.csv', 'zuzel.csv']:
    print(i +"," + str(for_individual_files(i)))

