####Reference: https://www.kaggle.com/danil328/ligthgbm-with-bayesian-optimization

from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold, KFold
import lightgbm as lgb
import numpy as np
import pandas as pd

data = pd.read_csv('train.csv')
target = data['target']
data.drop(['ID', 'target'], axis=1, inplace=True)

features = data.columns

test = pd.read_csv('test.csv')

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest MSE: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    clf_name = bayes_cv_tuner.estimator.__class__.__name__
    all_models.to_csv(clf_name+"_cv_results.csv")
    

bayes_cv_tuner = BayesSearchCV(
    estimator = lgb.LGBMRegressor(objective='regression', boosting_type='gbdt', subsample=0.6143), #colsample_bytree=0.6453, subsample=0.6143
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'num_leaves': (10, 100),      
        'max_depth': (0, 50),
        'min_child_samples': (0, 50),
        'max_bin': (100, 1000),
        'subsample_freq': (0, 10),
        'min_child_weight': (0, 10),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'scale_pos_weight': (1e-6, 500, 'log-uniform'),
        'n_estimators': (50, 150),
    },    
    scoring = 'neg_mean_squared_log_error', #neg_mean_squared_log_error
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    ),
    n_jobs = 1,
    n_iter = 100,   
    verbose = 0,
    refit = True,
    random_state = 42
)

# Fit the model
result = bayes_cv_tuner.fit(data[features], target, callback=status_print)

pred = bayes_cv_tuner.predict(test[features])
test['target'] = np.expm1(pred)
test[['ID', 'target']].to_csv('my_submission.csv', index=False, float_format='%.2f')
