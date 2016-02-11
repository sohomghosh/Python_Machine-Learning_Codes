#Removing Features with Low Variance NOT INCLUDED IN THIS VERSION OF SKLEARN
from sklearn.feature_selection import VarianceThreshold
X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(X)



#Univariate Feature Selection
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X, y = iris.data, iris.target
X.shape
X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
X_new.shape


#L1 based feature selection
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X.shape
X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)
X_new.shape


#Tree-based feature selection NOT WOrking
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X.shape
clf = ExtraTreesClassifier()
X_new = clf.fit(X, y).transform(X)
clf.feature_importances_
X_new.shape
