import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
train = pd.read_csv('numeric_train.csv')
# removing Salary
x = train.drop(['Salary'], axis=1)

#Keeping only numeric data
x = x._get_numeric_data()
y = train.Salary

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.4)



from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)

fp=open("score_lr.csv",'w')
fp.write("Actual,Predicted\n")
j=0
for i in y_test:
	fp.write(str(i)+","+str(y_pred[j])+"\n")
	j=j+1
fp.close()
#Evaluating Matrix
#from sklearn import metrices
#print metrices.accuracy_score(y_test,y_pred)