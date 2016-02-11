#Validating Curve NOT PRESENT IN THIS VERSION OF SKLEARN
import numpy as np
from sklearn.learning_curve import validation_curve
from sklearn.datasets import load_iris
from sklearn.linear_model import Ridge
np.random.seed(0)
iris = load_iris()
X, y = iris.data, iris.target
indices = np.arange(y.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
train_scores, valid_scores = validation_curve(Ridge(), X, y, "alpha", np.logspace(-7, 3, 3))
train_scores
valid_scores



#Learning Curves NOT PRESENT IN THIS VERSION OF SKLEARN
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC
train_sizes, train_scores, valid_scores = learning_curve(SVC(kernel='linear'), X, y, train_sizes=[50, 80, 110], cv=5)
train_sizes
train_scores
valid_scores