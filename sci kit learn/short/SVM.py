from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
print(digits.data)
digits.target #digits.target gives the ground truth for the digit dataset, that is the number corresponding to each digit image that we are trying to learn
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(digits.data[:-1], digits.target[:-1]) #We select this training set with the [:-1] Python syntax, which produces a new array that contains all but the last entry of digits.data
SVC(C=100.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
gamma=0.001, kernel='rbf', max_iter=-1, probability=False,
shrinking=True, tol=0.001, verbose=False)
clf.predict(digits.data[-1])



#ANOTHER EXAMPLE
from sklearn import svm
from sklearn import datasets
clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

#USING KERNALS
svc = svm.SVC(kernel='linear')
svc = svm.SVC(kernel='poly', degree=3)
svc = svm.SVC(kernel='rbf')