#Reference: https://www.kaggle.com/alexpengxiao/preprocessing-model-averaging-by-xgb-lgb-1-39
from scipy.stats import ks_2samp
THRESHOLD_P_VALUE = 0.01 #need tuned
THRESHOLD_STATISTIC = 0.3 #need tuned
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)
train.shape
