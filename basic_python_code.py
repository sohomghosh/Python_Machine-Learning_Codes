import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
train = pd.read_csv('numeric_train.csv')
# removing ID and Salary
X_train = train.drop(['Salary'], axis=1)

#Keeping only numeric data
X_train = X_train._get_numeric_data()
y_train = train.Salary

#print y_train.shape
#print X_train.shape
clf = LinearRegression()
clf = clf.fit(X_train, y_train)
print clf.intercept_
print clf.coef_
##print clf
##scores=cross_val_score(clf,X_train,y_train,cv=10,scoring='accuracy')

#test = pd.read_excel('test.xlsx')
# removing ID and Salary
#X_test = test.drop(['ID', 'Salary'], axis=1)
#Keeping only numeric data
#X_test = X_test._get_numeric_data()
#y_test = test.Salary

#r_sqr = clf.score(X_test, y_test)
#y_pred = clf.predict(X_test)
#mae = mean_absolute_error(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)
##print scores
#pd.DataFrame({'ID':test.ID,'Salary':y_pred}).to_excel('results.xlsx',index=False)