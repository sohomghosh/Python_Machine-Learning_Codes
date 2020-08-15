#Reference:   https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/
#Covariance Matrix Calculation: https://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm

#REMEMBER DO NOT INCLUDE 'Y' i.e. Depedent Variable as input to PCA
#It is a good practise to scale the data before running a PCA
#Using package PCA
from numpy import array
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
scaler = StandardScaler()
scaler.fit(A)
A_scaled = scaler.transform(A_scaled)
# create the PCA instance
pca = PCA(2) #2 is the number of principal components we want
# fit on data
pca.fit(A_scaled)
# access values and vectors
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.explained_variance_)
# transform data
B = pca.transform(A_scaled)
print(B) #B is the final transformed data which is to be used


#################################################################################################################


from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
# define a matrix
A = array([[1, 2], [3, 4], [5, 6]])
print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)

