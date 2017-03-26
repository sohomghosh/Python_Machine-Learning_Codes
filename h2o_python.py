import pandas as pd
import numpy as np
#from sklearn import preprocessing, model_selection, metrics, ensemble
#from sklearn.preprocessing import LabelEncoder
import h2o

# initialize an h2o cluster
h2o.init()
#h2o.init(nthreads = -1,max_mem_size = "6G") 
h2o.connect()
train = h2o.import_file("train_indessa.csv")
test = h2o.import_file("test_indessa.csv")

# all values of r under .8 are assigned to 'train_split' (80 percent)
train['batch_enrolled']=train['batch_enrolled'].asfactor()

for i in ['grade','sub_grade','emp_title','emp_length','home_ownership','verification_status','pymnt_plan','desc','purpose','title','zip_code','addr_state','initial_list_status','application_type','verification_status_joint','last_week_pay']:
	train[i]=train[i].asfactor()
train["loan_status"]=train["loan_status"].asfactor()

r = train.runif()   
train_split = train[r  < 0.8]
valid_split = train[r >= 0.8]
features = list(np.setdiff1d(train.names, ['emp_title', 'desc','purpose','title','zip_code','addr_state','loan_status']))



#DEEP LEARNING
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
model_dl = H2ODeepLearningEstimator(
        distribution="multinomial",
        activation="RectifierWithDropout",
        hidden=[100,200,100],
        input_dropout_ratio=0.2, 
        sparse=True, 
        l1=1e-5, 
        epochs=100)

model_dl.train(
        x= features, 
        y="loan_status", 
        training_frame=train_split, 
        validation_frame=valid_split)
model_dl.params
print(model_dl)



#GBM
from h2o.estimators.gbm import H2OGradientBoostingEstimator
model_gbm = H2OGradientBoostingEstimator(distribution='bernoulli',
                                    ntrees=100,
                                    max_depth=4,
                                    learn_rate=0.1)
model_gbm.train(x=features, y="loan_status", training_frame=train_split, validation_frame=valid_split)
print(model_gbm)



#GBM with cross validation
cvmodel = H2OGradientBoostingEstimator(distribution='bernoulli',
                                       ntrees=100,
                                       max_depth=4,
                                       learn_rate=0.1,
                                       nfolds=5)
cvmodel.train(x=features, y="loan_status", training_frame=train)
print(cvmodel)


#GBM - Grid Search
from h2o.grid.grid_search import H2OGridSearch
ntrees_opt = [5,50,100]
max_depth_opt = [2,3,5]
learn_rate_opt = [0.1,0.2]
hyper_params = {'ntrees': ntrees_opt, 
                'max_depth': max_depth_opt,
                'learn_rate': learn_rate_opt}
gs = H2OGridSearch(H2OGradientBoostingEstimator, hyper_params = hyper_params)
gs.train(x=features, y="loan_status", training_frame=train_split, validation_frame=valid_split)
print(gs)

for g in gs:
    print(g.model_id + " auc: " + str(g.auc()))



#Random Forest
from h2o.estimators.random_forest import H2ORandomForestEstimator
model_rf = H2ORandomForestEstimator(ntrees=250, max_depth=30)

model_rf.train(x=features,
               y="loan_status",
               training_frame  =train_split,
               validation_frame=valid_split)
print(model_rf)


#GLM
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
model_glm = H2OGeneralizedLinearEstimator(Lambda=[1e-5], family="poisson")
    
model_glm.train(x=features,
                y="loan_status",
                training_frame  =train_split,
                validation_frame=test_split)




pred = model.predict(test)
pred.head()
submit_pred= pred[:,1]
submit_pred.head()
submission_dataframe =(test[:,'member_id']).cbind(submit_pred)
submission_dataframe.set_name(1,"loan_status")
h2o.h2o.export_file(submission_dataframe, path ="submission_rf_1.csv")
