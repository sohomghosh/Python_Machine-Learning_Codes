#ACCURACY measure CLASSIFICATION
import mlpy
t = [3,2,3,3,3,1,1,1]
p = [3,2,1,3,3,2,1,1]
mlpy.error(t, p)
mlpy.accuracy(t, p)


#Sensitivity, Specitivity, AUC
import mlpy
t = [1, 1, 1,-1, 1,-1,-1,-1]
p = [1,-1, 1, 1, 1,-1, 1,-1]
mlpy.error_p(t, p)
mlpy.error_n(t, p)
mlpy.sensitivity(t, p)
mlpy.specificity(t, p)
mlpy.ppv(t, p)
mlpy.npv(t, p)
mlpy.mcc(t, p)
p = [2.3,-0.4, 1.6, 0.6, 3.2,-4.9, 1.3,-0.3]
mlpy.auc_wmw(t, p)
p = [2.3,0.4, 1.6, -0.6, 3.2,-4.9, -1.3,-0.3]
mlpy.auc_wmw(t, p)


#Mean Squared Error REGRESSION
import mlpy
t = [2.4,0.4,1.2,-0.2,3.3,-4.9,-1.1,-0.1]
p = [2.3,0.4,1.6,-0.6,3.2,-4.9,-1.3,-0.3]
mlpy.mse(t, p)
