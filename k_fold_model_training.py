import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
import xgboost as xgb

folds = KFold(n_splits=5,shuffle=False)

#test_x : is the test dataframe
#train_data : is the trainining dataframe

#sub_preds = np.zeros(test_X.shape[0])
oof_preds = np.zeros(train_data.shape[0])

params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 4, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}

nrounds = 100
watchlist = [(dtrain, 'train')]

for trn_idx, val_idx in folds.split(train_data):
    tr_X, val_X = train_data[features].iloc[trn_idx], train_data[features].iloc[val_idx]
    tr_y, val_y = train_data['label'].iloc[trn_idx], train_data['label'].iloc[val_idx]
    dtrain = xgb.DMatrix(tr_X, tr_y, missing=np.nan)
    bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)
    
    # Predict Out Of Fold and Test targets
    # Using lgb.train, predict will automatically select the best round for prediction
    oof_preds[val_idx] = bst.predict(val_X)
    #sub_preds += bst.predict(test_X[features]) / folds.n_splits
    # Display current fold score
    print(roc_auc_score(val_y, oof_preds[val_idx]))
# Display Full OOF score (square root of a sum is not the sum of square roots)
print('Full Out-Of-Fold score : %9.6f'% (roc_auc_score(train_data['label'], oof_preds)))
