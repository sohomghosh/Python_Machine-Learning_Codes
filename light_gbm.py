import pandas as pd
import lightgbm as lgb

X_train = <pandas dataframe>
y_train = <pandas dataframe>
X_test = <pandas dataframe>

gbm = lgb.LGBMClassifier(n_estimators=2900, max_depth=3, subsample=0.7, colsample_bytree= 0.7)
gbm = gbm.fit(X_train, y_train)
y_test = gbm.predict_proba(X_Test)
