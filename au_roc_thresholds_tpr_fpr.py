#Reference: https://github.com/abulbasar/machine-learning/blob/master/Scikit%20-%2006%20Text%20Processing.ipynb
fpr, tpr, thresholds = metrics.roc_curve(X_valid['label'], valid_preds)
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.plot([0,1], [0,1], ls = "--", color = "k")
plt.xlabel("False Postive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve, auc: %.4f" % auc);



#Reference: https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
import pylab as pl
fpr, tpr, thresholds = roc_curve(X_valid['label'], valid_preds)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)
####################################
# The optimal cut off would be where tpr is high and fpr is low
# tpr - (1-fpr) is zero or near to zero is the optimal cut off point
####################################
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'])
pl.plot(roc['1-fpr'], color = 'red')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
pl.show()
