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

