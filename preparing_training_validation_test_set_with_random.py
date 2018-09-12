import random
train_ids = random.sample(list(train.index),int(.8*len(train.index)))

train_new = train[train.index.isin(train_ids)]
valid = train[~train.index.isin(train_ids)]




#Other methods
features_col=['TV','Radio','Newspaper']
X=data[features_col]
X.head()
print type(X)
print X.shape
y=data['Sales']
y.head()
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,random_state=1)
