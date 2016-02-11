import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
from sklearn import linear_model
logistic = linear_model.LogisticRegression(C=1e5)
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
logistic.fit(iris_X_train, iris_y_train)