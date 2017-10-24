#Encode all categorical features  {Typecaste Typecaste to float: Country, TrafficType, Device, Browser, OS, RefererUrl, publisherId,subPublisherId}
#Missing value treatment
#**11) Rule if fraud then ConversionPayout = 0

#hadoop fs -put train_clean /index/sohom_exeriment/av_ctr_sep17
#hadoop fs -put test_clean /index/sohom_exeriment/av_ctr_sep17

#/opt/spark-2.1.0-bin-hadoop2.7/bin/./pyspark --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3
#/opt/sparkling-water-2.1.14/bin/./pysparkling --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType,StructField,IntegerType
import h2o
from pysparkling import *
import numpy as np
from h2o.estimators.gbm import H2OGradientBoostingEstimator


#########################################################Train########################################################
h2oContext = H2OContext.getOrCreate(spark)

#Read data from hadoop as spark dataframe and transform it to h2o data frame
train=spark.read.csv('hdfs://hadoop-master:9000/index/sohom_exeriment/av_ctr_sep17/train_clean',header=True)
train_h2o=h2oContext.as_h2o_frame(train)

#Load local csv data to h2o pyspark cluster
##train_h2o=h2o.upload_file('/index/sohom_exeriment/av_ctr_sep17/train_clean')


#Load csv file in hadoop to h2o pyspark cluster
##train_h2o=h2o.import_file(path = "hdfs://hadoop-master:9000/index/sohom_exeriment/av_ctr_sep17/train_clean")
#Didn't work: Error: water.parser.ParseDataset$H2OParseException: Exceeded categorical limit on columns [UserIp].   Consider reparsing these columns as a string. stacktrace: 


splits = train_h2o.split_frame(ratios=[0.8])

train_use_h2o=splits[0]
valid_use_h2o=splits[1]

cat_cols = ["Country", "TrafficType", "Device", "Browser", "OS"]#"UserIp","RefererUrl", "publisherId","subPublisherId"

for col in cat_cols:
    train_use_h2o[col] = train_use_h2o[col].asfactor()
    valid_use_h2o[col] = valid_use_h2o[col].asfactor()


predictor_columns = train_h2o.drop(["ID","ConversionPayOut"]).col_names
response_column = "ConversionPayOut"

                                           max_depth    = 3,
                                           learn_rate   = 0.1,
                                           distribution = "bernoulli"
                                           )


gbm_model.train(x                = predictor_columns,
                y                = response_column,
                training_frame   = train_use_h2o,
                validation_frame = valid_use_h2o
                )


print(gbm_model)

model=gbm_model
h2o.saveModel(model, dir = "hdfs://hadoop-master:9000/index/sohom_exeriment/av_ctr_sep17", name = "model_1",force = True)
h2o.shutdown(prompt=False)


#########################################################Test########################################################
test=spark.read.csv('hdfs://hadoop-master:9000/index//sohom_exeriment/av_ctr_sep17/test_clean',header=True)

h2oContext = H2OContext.getOrCreate(spark)

test_h2o=h2oContext.as_h2o_frame(train)
cat_cols = ["Country", "TrafficType", "Device", "Browser", "OS", "RefererUrl", "publisherId","subPublisherId","UserIp"]

for col in cat_cols:
	test_h2o[col] = test_h2o[col].asfactor()

h2o.loadModel("/h2o.shutdown(prompt=False)/index/sohom_exeriment/av_ctr_sep17/model_1", conn = h2o.getConnection())

pred = model.predict(test_h2o)
pred.head()
submit_pred= pred[:,1]
submit_pred.head()
submission_dataframe =(test_h2o[:,'ID']).cbind(submit_pred)
submission_dataframe.set_name(1,"ConversionPayOut")
h2o.h2o.export_file(submission_dataframe, path ="submission_1.csv")

h2o.shutdown(prompt=False)





'''
def extract_ip_info(ip):
	ip=str(ip).strip()
	try:
		sp=ip.split('.')
		return int(sp[0]),int(sp[1]),int(sp[2]),int(sp[3])
	except:
		return np.nan,np.nan,np.nan,np.nan


schema_ip_ex = StructType([
    StructField("ip1", IntegerType(), False),
    StructField("ip2", IntegerType(), False),
    StructField("ip3", IntegerType(), False),
    StructField("ip4", IntegerType(), False)
])

udf_extractip=udf(extract_ip_info,schema_ip_ex)

train=train.withColumn('ips_new',udf_extractip("UserIp"))
test=test.withColumn('ips_new',udf_extractip("UserIp"))

train = train.drop('UserIp')
test = test.drop('UserIp')


stringIndexer = StringIndexer(inputCol="currency", outputCol="currencyindexed")
model = stringIndexer.fit(train_test)
td = model.transform(train_test)
encoder = OneHotEncoder(inputCol="currencyindexed", outputCol="currency_features")
train_test=encoder.transform(td)
'''
