#Reference: https://github.com/HackerEarth-Challenges/ML-Challenge-3/blob/master/CatBoost_Starter.ipynb

#pip3 install catboost
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

#...other lines of <code> here...#

trainX = sampled_train[cols_to_use]
trainY = sampled_train['click']

X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.5)

# catboost accepts categorical variables as indexes
cat_cols = [0,1,2,4,6,7,8]

#cols_to_use refers to the columns to be used as features
cols_to_use = list(set(train.columns) - set(['ID','datetime','click']))

model = CatBoostClassifier(depth=10, iterations=10, learning_rate=0.1, eval_metric='AUC', random_seed=1)

model.fit(X_train,y_train,cat_features=cat_cols,eval_set = (X_test, y_test),use_best_model = True)
pred = model.predict_proba(test[cols_to_use])[:,1]

sub = pd.DataFrame({'ID':test['ID'],'click':pred})
sub.to_csv('cb_sub1.csv',index=False)

