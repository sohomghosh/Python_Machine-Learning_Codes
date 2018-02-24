#Loading Modules
import numpy as np
import mlpy
import matplotlib.pyplot as plt # required for plotting

#Loading Dataset
iris = np.loadtxt('iris.csv', delimiter=',')
x, y = iris[:, :4], iris[:, 4].astype(np.int) # x: (observations x attributes) matrix, y: classes (1: setosa, 2: versicolor, 3: virginica)
x.shape
y.shape


########################PCA USING SKLEARN#########################
pca = PCA(n_components=2)
pca.fit(vecs_of_nodes)
z = pca.transform(vecs_of_nodes)
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)
plt.set_cmap(plt.cm.Paired)
fig1 = plt.figure(1)
title = plt.title("PCA on vecs of nodes")
plot = plt.scatter(z[:, 0], z[:, 1])
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")

########################PCA USING MLPY#############################
#Dimensionality reduction by Principal Component Analysis (PCA)
pca = mlpy.PCA() # new PCA instance
pca.learn(x) # learn from data
z = pca.transform(x, k=2) # embed x into the k=2 dimensional subspace
z.shape


#Plot the principal components:
plt.set_cmap(plt.cm.Paired)
fig1 = plt.figure(1)
title = plt.title("PCA on iris dataset")
plot = plt.scatter(z[:, 0], z[:, 1], c=y)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")
plt.show()


#Learning by Kernel Support Vector Machines (SVMs) on principal components:
linear_svm = mlpy.LibSvm(kernel_type='linear') # new linear SVM instance
linear_svm.learn(z, y) # learn from principal components

#For plotting purposes, we build the grid where we will compute the predictions (zgrid):
xmin, xmax = z[:,0].min()-0.1, z[:,0].max()+0.1
ymin, ymax = z[:,1].min()-0.1, z[:,1].max()+0.1
xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
zgrid = np.c_[xx.ravel(), yy.ravel()]

#Now we perform the predictions on the grid. The pred() method returns the prediction for each point in zgrid:
yp = linear_svm.pred(zgrid)

#Plot the predictions:
plt.set_cmap(plt.cm.Paired)
fig2 = plt.figure(2)
title = plt.title("SVM (linear kernel) on principal components")
plot1 = plt.pcolormesh(xx, yy, yp.reshape(xx.shape))
plot2 = plt.scatter(z[:, 0], z[:, 1], c=y)
labx = plt.xlabel("First component")
laby = plt.ylabel("Second component")
limx = plt.xlim(xmin, xmax)
limy = plt.ylim(ymin, ymax)
plt.show()




