#Reference: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


data=pd.read_csv("data_churn.csv")

lbl = LabelEncoder()
lbl.fit(list(data['X1'].values))
data['Feature1_encoded'] = lbl.transform(list(data['Feature1'].values))

features = np.setdiff1d(data.columns, ['label','id'])

#If any other feature is remaining which is int but of string type -->> converting them into int
for i in features:
	if data[i].dtypes=='object':
		data[i]=pd.to_numeric(pd.Series(data[i]),errors='coerce')
		print(i)


data_use=data[features]
X_train=data.sample(frac=0.2, replace=False)
X_valid=pd.concat([data, X_train]).drop_duplicates(keep=False)


######################## XG-BOOST WITHOUT CV AND GRID SEARCH################################################################
#nrounds = 260
#watchlist = [(dtrain, 'train')]
model=xgb.XGBClassifier(objective="binary:logistic",nthread= 4, silent= 0, max_depth= 5, min_child_weight=1,subsample=0.85,colsample_bytree=0.8,max_delta_step=0,gamma=0,seed=0,scale_pos_weight=1,base_score=0.5)

model.fit(X_train[features],X_train['label'])
valid_preds= model.predict(X_valid[features])

print(mean_squared_error(list(X_valid['label']), valid_preds))
#0.367

fpr, tpr, thresholds = metrics.roc_curve(X_valid['label'], valid_preds)
print(metrics.auc(fpr, tpr))
#0.54

print(model.feature_importances_)
print(list(model.feature_importances_))
print(list(features))

######################## XG-BOOST WITH CV AND GRID SEARCH################################################################
def modelfit(alg, dtrain, features,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[features].values, label=dtrain['label'].values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds)#, show_progress=False)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(dtrain[features], dtrain['Renewed_encoded'],eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain[features])
    dtrain_predprob = alg.predict_proba(dtrain[features])[:,1]
        
    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['label'].values, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['label'], dtrain_predprob))
                    
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')


model=xgb.XGBClassifier(objective="binary:logistic",nthread= 4, silent= 0, max_depth= 5, min_child_weight=1,subsample=0.85,colsample_bytree=0.8,max_delta_step=0,gamma=0,seed=0,scale_pos_weight=1,base_score=0.5)
param_test1 = {'max_depth':[3,10,2],'min_child_weight':[1,6,2]}
xgb1 = xgb.XGBClassifier(learning_rate =0.1, n_estimators=100, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
modelfit(xgb1, X_train, features)

gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27),  param_grid = param_test1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
gsearch1.fit(X_train[features],X_train["label"])
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
