#######LEFT
##Add param grid: from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, tfidf, word2vec

#AFTER#Make .py for spark-submit
#make the whole model run
#Machine Learning models
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassificationModel
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassificationModel
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.MultilayerPerceptronClassificationModel

#./pyspark --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3

import numpy as np
from pyspark.sql.types import StructType,StructField,LongType,StringType,TimestampType
from pyspark.sql.types import StructType,StructField,LongType,StringType,TimestampType,ArrayType
from pyspark .sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import col, split, udf
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.classification import RandomForestClassifier as RF
import time
from pyspark.ml.feature import VectorIndexer, VectorAssembler, SQLTransformer
from pyspark.ml import Pipeline
from pyspark.sql.functions import lit
from pyspark.ml.linalg import SparseVector, DenseVector 

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import GBTClassifier

#LOCAL
train = spark.read.csv('file:///home//sohom/Desktop/HE_ML2/data/train.csv',mode="DROPMALFORMED",header=True)
test = spark.read.csv('file:///home//sohom/Desktop/HE_ML2/data/test.csv',mode="DROPMALFORMED",header=True)

#SERVER
train = spark.read.csv('hdfs://hadoop-master:9000/index/HE_ML2/train.csv',mode="DROPMALFORMED",header=True)
test = spark.read.csv('hdfs://hadoop-master:9000/index/HE_ML2/test.csv',mode="DROPMALFORMED",header=True)

train = train.drop('backers_count')

train.columns
#project_id, name, desc, goal, keywords, disable_communication, country, currency, deadline, state_changed_at, created_at, launched_at, final_status

test.columns
#'project_id', 'name', 'desc', 'goal', 'keywords', 'disable_communication', 'country', 'currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at'

train.count()
#102482

test.count()
#60135

#Adding a null column to test
test=test.withColumn('final_status',lit(None).cast(StringType()))

train_test=train.union(test)

#project_id [REMOVE]
#name [REMOVE]

#USE COLUMNS: goal (as it is), keywords (CV), disable_communication (ENCODED), country (ENCODED), currency (ENCODED), time difference in days


#Value counts
train.groupby('disable_communication').count().show()


train_test=train_test.select(train_test.project_id,train_test.name,train_test.desc,train_test.goal,train_test.keywords, F.when(train_test.disable_communication =='false', 0).when(train_test.disable_communication =='true', 0).otherwise(np.nan).alias('disable_communication_encoded'),train_test.country,train_test.currency,train_test.deadline,train_test.state_changed_at,train_test.created_at,train_test.launched_at,train_test.final_status)

'''
+---------------------+------+                                                  
|disable_communication| count|
+---------------------+------+
|                false|102171|
|                 true|   311|
+---------------------+------+
'''


stringIndexer = StringIndexer(inputCol="country", outputCol="countryindexed")
model = stringIndexer.fit(train_test)
td = model.transform(train_test)
encoder = OneHotEncoder(inputCol="countryindexed", outputCol="country_features")
train_test=encoder.transform(td)


'''
+-------+-----+
|country|count|
+-------+-----+
|     NL|  639|
|     AU| 1778|
|     CA| 3528|
|     GB| 8344|
|     DE|    1|
|     US|87254|
|     SE|  225|
|     NZ|  329|
|     IE|  103|
|     NO|  103|
|     DK|  178|
+-------+-----+
'''

train.groupby('currency').count().show()


stringIndexer = StringIndexer(inputCol="currency", outputCol="currencyindexed")
model = stringIndexer.fit(train_test)
td = model.transform(train_test)
encoder = OneHotEncoder(inputCol="currencyindexed", outputCol="currency_features")
train_test=encoder.transform(td)


'''
|currency|count|
+--------+-----+
|     DKK|  178|
|     NZD|  329|
|     GBP| 8344|
|     CAD| 3528|
|     EUR|  743|
|     NOK|  103|
|     AUD| 1778|
|     USD|87254|
|     SEK|  225|
+--------+-----+
'''


train.groupby('final_status').count().show()
'''
+------------+-----+
|final_status|count|
+------------+-----+
|           0|69629| #0.679
|           1|32853| #0.320
+------------+-----+
'''

#Text columns
#desc,keywords [TEXT]


train_test=train_test.withColumn('keyword_features',split(col('keywords'),'-'))
cv = CountVectorizer(inputCol="keyword_features", outputCol="keyword_features_cv")
model=cv.fit(train_test)
train_test = model.transform(train_test)
train_test.show(truncate=False)




train.columns
#['project_id', 'name', 'desc', 'goal', 'keywords', 'disable_communication_encoded', 'country', 'currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at', 'final_status', 'countryindexed', 'country_features', 'currencyindexed', 'currency_features', 'keyword_features', 'keyword_features_cv', 'diff_statechange_deadline', 'diff_created_deadline', 'diff_launched_deadline', 'diff_statechange_launched']

train_test=train_test.withColumn('diff_statechange_deadline',(train_test.state_changed_at-train_test.deadline)/86400)
train_test=train_test.withColumn('diff_created_deadline',(train_test.deadline-train_test.created_at)/86400)
train_test=train_test.withColumn('diff_launched_deadline',(train_test.deadline-train_test.launched_at)/86400)
train_test=train_test.withColumn('diff_statechange_launched',(train_test.state_changed_at-train_test.launched_at)/86400)

train_test.columns
#['project_id', 'name', 'desc', 'goal', 'keywords', 'disable_communication_encoded', 'country', 'currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at', 'countryindexed', 'country_features', 'currencyindexed', 'currency_features', 'keyword_features', 'keyword_features_cv', 'diff_statechange_deadline', 'diff_created_deadline', 'diff_launched_deadline', 'diff_statechange_launched']


train_test_use=train_test.select(train_test.project_id,train_test.goal.cast('float').alias('goal'),train_test.disable_communication_encoded,train_test.country_features,train_test.currency_features,train_test.keyword_features_cv,train_test.diff_statechange_deadline,train_test.diff_created_deadline,train_test.diff_launched_deadline,train_test.diff_statechange_launched,train_test.final_status.cast('float'))

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType()))(col)


cols_take=['project_id','goal','disable_communication_encoded','diff_statechange_deadline', 'diff_created_deadline', 'diff_launched_deadline', 'diff_statechange_launched','currency_features','final_status','keyword_features_cv']

train_test_use=train_test_use.withColumn("country_individual_features", to_array(col("country_features"))).select(cols_take+[col("country_individual_features")[i] for i in range(20)])

train_test_use=train_test_use.withColumn("currency_individual_features",to_array(col("currency_features"))).select(train_test_use.columns+[col("currency_individual_features")[i] for i in range(12)])

train_test_use=train_test_use.withColumn("keyword_individual_features_cv",to_array(col("keyword_features_cv"))).select(train_test_use.columns+[col("keyword_individual_features_cv")[i] for i in range(126)])

train_test_use=train_test_use.drop('currency_features').drop('keyword_features_cv')

cols_now=train_test_use.columns
cols_now.remove('project_id')

#####------WORKING
#####cols_now=['goal','disable_communication_encoded','diff_statechange_deadline', 'diff_created_deadline', 'diff_launched_deadline', 'diff_statechange_launched']

assembler_features = VectorAssembler(inputCols=cols_now, outputCol='features')
labelIndexer = StringIndexer(inputCol='final_status', outputCol="label")
pipeline = Pipeline(stages=[assembler_features,labelIndexer])

##########################################Preparing Test Data###########################################################################

########################################################################################################################################

train_use=train_test_use.limit(102482)#train.count() is 102482
test_use=train_test_use.join(train_use,train_test_use.project_id==train_use.project_id,how='leftanti')

train_use=train_use.drop('project_id')
test_use=test_use.drop('project_id','final_status')

###################-------------RF-------------------------------

train_use=train_use.na.drop() #Droping na values
#train_use=train_test_use.limit(2482)
trainingData = pipeline.fit(train_use).transform(train_use)
rf = RF(labelCol='label', featuresCol='features',numTrees=20)
fit = rf.fit(trainingData)
#https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier

####Predict
pipeline=Pipeline(stages=[assembler_features])
test_use=test_use.na.drop()
testData=pipeline.fit(test_use).transform(test_use)
transformed = fit.transform(testData)
#transformed has 3 notable columns:  rawPrediction|probability|prediction|
predictions=transformed.select(transformed.prediction)
predictions.write.csv(path="/index/predictions_rf.csv")

#-----------GBM---------------------------------
#https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-tree-classifier
# Train a GBT model.
gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

#Train model.  This also runs the indexers.
model = gbt.fit(trainingData)

###Failed to execute user defined function($anonfun$4: (string) => double)
###Caused by: org.apache.spark.SparkException: Unseen label: null.

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("prediction", "label", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))

gbtModel = model.stages[2]
print(gbtModel)  # summary only


#-----------MLP--------------------------------
#https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier
#%%ERROR%% #Failed to execute user defined function($anonfun$4: (string) => double)

# specify layers for the neural network:
# input layer of size 4 (features), two intermediate of size 5 and 4
# and output of size 3 (classes)
layers = [4, 5, 4, 3]

# create the trainer and set its parameters
trainer = MultilayerPerceptronClassifier(maxIter=100, layers=layers, blockSize=128, seed=1234,labelCol="label", featuresCol="features")

model = trainer.fit(trainingData)
result = model.transform(testData)

predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))















#-----------AVOIDING desc because it is already present in keywords---------------------------------------
documents=train.select(['project_id','desc'])
df=documents.withColumn('desc_features',split(col('desc'),' '))
cv = CountVectorizer(inputCol="desc_features", outputCol="desc_features_cv")
model=cv.fit(df)
result_desc = model.transform(df)
result_desc.show(truncate=False)
#---------------------------------------------------------------

#----------TFIDF---NOT SUCCESSFUL----------------
https://stackoverflow.com/questions/35769489/adding-the-resulting-tfidf-calculation-to-the-dataframe-of-the-original-document

from pyspark.ml.feature import HashingTF as MLHashingTF
from pyspark.ml.feature import IDF as MLIDF
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import udf
from pyspark.sql.functions import col, split


'''
documents=train.select(['project_id','desc'])
df=documents.withColumn("word_features",split(col('desc'),' '))
htf = MLHashingTF(numFeatures=30,inputCol="word_features", outputCol="tf")
tf = htf.transform(df)
tf.show(truncate=False)
idf = MLIDF(inputCol="tf", outputCol="idf")
tfidf = idf.fit(tf).transform(tf)
tfidf.show(truncate=False)
'''
#------------------------------------------------

'''
def words_df(desc_col):
	return str(desc_col).split(' ')

udf_words_df = udf(words_df, StringType())


#df = documents.rdd.map(lambda x : x.desc.split(" ")).toDF()
#.withColumnRenamed("_1","train_id").withColumnRenamed("_2","features"))
df=documents.withColumn('word_features',udf_words_df('desc'))
df.printSchema()
df.withColumn('word_features_new',df.word_features.cast(ArrayType()))



documents.withColumn("word_features",split(col('desc'),' ').cast(ArrayType()))

#df=documents.withColumn('word_features',udf_words_df('desc').cast(ArrayType()))
'''





'''
#train_use=train.select(train.goal.cast('float').alias('goal'),train.disable_communication_encoded,train.country_features,train.currency_features,train.keyword_features_cv,train.diff_statechange_deadline,train.diff_created_deadline,train.diff_launched_deadline,train.diff_statechange_launched,train.final_status.cast('float'))


#Create Features
#deadline - state_changed_at
#deadline - created_at
#deadline - launched_at
#created_at - launched_at
#state_changed_at - launched_at


#86400
# train=train.withColumn('diff_statechange_deadline',(train.state_changed_at-train.deadline)/86400)
# train=train.withColumn('diff_created_deadline',(train.deadline-train.created_at)/86400)
# train=train.withColumn('diff_launched_deadline',(train.deadline-train.launched_at)/86400)
# train=train.withColumn('diff_statechange_launched',(train.state_changed_at-train.launched_at)/86400)


#documents=train.select(['project_id','keywords'])
# train=train.withColumn('keyword_features',split(col('keywords'),'-'))
# cv = CountVectorizer(inputCol="keyword_features", outputCol="keyword_features_cv")
# model=cv.fit(train)
# train = model.transform(train)
# train.show(truncate=False)


# stringIndexer = StringIndexer(inputCol="currency", outputCol="currencyindexed")
# model = stringIndexer.fit(train)
# td = model.transform(train)
# encoder = OneHotEncoder(inputCol="currencyindexed", outputCol="currency_features")
# train=encoder.transform(td)


# stringIndexer = StringIndexer(inputCol="country", outputCol="countryindexed")
# model = stringIndexer.fit(train)
# td = model.transform(train)
# encoder = OneHotEncoder(inputCol="countryindexed", outputCol="country_features")
# train=encoder.transform(td)

#train=train.select(train.project_id,train.name,train.desc,train.goal,train.keywords, F.when(train.disable_communication =='false', 0).when(train.disable_communication =='true', 0).otherwise(np.nan).alias('disable_communication_encoded'),train.country,train.currency,train.deadline,train.state_changed_at,train.created_at,train.launched_at,train.final_status)



train.withColumn('diff_deadline_statechange',(datetime.fromtimestamp(train['deadline'])-datetime.fromtimestamp(train['state_changed_at'])).days).show()

import time

def time_convert(ti):
	#yr=time.strftime('%Y', time.localtime(ti))
	#mn=time.strftime('%Y', time.localtime(ti))
	DAY = 86400 # POSIX day - the exact value
	days_ago = (time.time() - sub.created_utc) // DAY

time.strftime('%Y', time.localtime(1347517370))

from pyspark.ml.feature import CountVectorizer
#cv = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
documents=train.select(['project_id','keywords'])
df=documents.withColumn('keyword_features',split(col('keywords'),'-'))
cv = CountVectorizer(inputCol="keyword_features", outputCol="keyword_features_cv")
model=cv.fit(df)
result = model.transform(df)
result.show(truncate=False)



from pyspark.mllib.linalg import SparseVector, DenseVector
train_use.withColumn('country_features_dense',DenseVector(col('country_features'))).show()

'''