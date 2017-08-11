#class sklearn.feature_extraction.DictVectorizer(dtype=<type 'numpy.float64'>, separator='=', sparse=True, sort=True)

from sklearn.feature_extraction import DictVectorizer
v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)
X



#sklearn.model_selection.cross_val_predict(estimator, X, y=None, groups=None, cv=None, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', method='predict')

from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict
diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
lasso = linear_model.Lasso()
y_pred = cross_val_predict(lasso, X, y)



#http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html

from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
skf = StratifiedKFold(n_splits=2)
skf.get_n_splits(X, y)
#2

print(skf)  
#StratifiedKFold(n_splits=2, random_state=None, shuffle=False)


for train_index, test_index in skf.split(X, y):
	print("TRAIN:", train_index, "TEST:", test_index)
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]


#TRAIN: [1 3] TEST: [0 2]
#TRAIN: [0 2] TEST: [1 3]
