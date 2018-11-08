#Source: https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/
#https://spark.apache.org/docs/latest/ml-features.html
#Source: http://spark.apache.org/docs/latest/api/python/pyspark.html

#Configurations: https://spark.apache.org/docs/2.2.0/configuration.html

#Other Links
#https://spark.apache.org/docs/latest/api/python/pyspark.html
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.MultilayerPerceptronClassificationModel.layers
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.RandomForestClassificationModel
#http://spark.apache.org/docs/latest/api/python/pyspark.ml.html#pyspark.ml.classification.GBTClassificationModel

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
from pyspark.sql.functions import explode

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import GBTClassifier

from pyspark.sql.functions import max,min

from pyspark import SparkContext
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import when

sc = SparkContext()

sqlContext = SQLContext(sc)

#Creating DataFrame from RDD
from pyspark.sql import Row
l = [('Ankit',25),('Jalfaizy',22),('saurabh',20),('Bala',26)]
rdd = sc.parallelize(l)
people = rdd.map(lambda x: Row(name=x[0], age=int(x[1])))
people.collect()
schemaPeople = sqlContext.createDataFrame(people)
#Alternate way:::       schemaPeople = people.toDF(['name','age'])


#For databricks related packages
#./bin/pyspark --packages com.databricks:spark-csv_2.10:1.3.0

#Before Spark 1.4
train = sqlContext.load(source="com.databricks.spark.csv", path = 'PATH/train.csv', header = True,inferSchema = True)
test = sqlContext.load(source="com.databricks.spark.csv", path = 'PATH/test-comb.csv', header = True,inferSchema = True)

#Current Spark 2.1 and ...
from pyspark .sql import SparkSession
spark = SparkSession.builder.master("yarn").getOrCreate()
df = spark.read.csv('hdfs://hadoop-master:9000/index/train.csv',mode="DROPMALFORMED")

#From local
from pyspark.sql.types import StructType,StructField,LongType,StringType,TimestampType
schema=StructType([StructField('', LongType(), True), StructField('col1', LongType(), True), StructField('col2', StringType(), True), StructField('col3', StringType(), True),StructField('col4',TimestampType(),True),StructField('col5',TimestampType(),True),StructField('col6',StringType(),True)])
df = spark.read.csv('file:///index/data_extract_restart2_without_cert/data_refined.csv',,mode="DROPMALFORMED"),schema=schema)

#Creating UDF
def dict(sk):
  new_sk=sk.replace(',','|')#replacing comma by pipe in column col2 and putting the result in column named new_column_name
	return new_sk


udf_dict = udf(dict, StringType())

df.withColumn('new_column_name', udf_dict("col2")).write.csv(path="/index/skill_clean_v3")#col2 is the column to be changed

#Executing SQL queries
df.createOrReplaceTempView("data")
sqlDF = spark.sql("SELECT * FROM data")
sqlDF.show(10)

df.printSchema()
df.head(5)

df.show(2,truncate= True)

#substring : extrcat hour from datetime i.e. from 2018-01-10 22:00:02 to 22
df.withColumn('ts', F.split(F.col('dateTime').substr(11,13), ':')[0])

#Extract 123 from 123.77 [remove part after decimal]
df = df.withColumn("clnId",F.split("clnid",':')[0])

#Number of columns
len(df.columns)

#Names of columns
df.columns

#Summary statistics of each columns
df.describe().show()

#Summary statistics of a particular column
df.describe('particular_column_name').show()

#Select particular column(s)
df.select('User_ID','Age').show(5)

#Distinct number of items in a column 'Item'; Unique number of items in a column 'Item'
df.select('Item').distinct().count()

#Pair wise frequency of categorical columns; pivot like; Rows to columns
df.crosstab('Age', 'Gender').show()

#Drop duplicates
train.select('Age','Gender').dropDuplicates().show()

#Drop duplicates
df.dropDuplicates(['name', 'height']).show()
#Drop na
df.dropna()
df.dropna().count()

#Replace na by zero
df.fillna(0)


#Filter - Select entries for which Purchase is > 15000
df.filter(df.Purchase > 15000)

df.filter((df.Purchase > 15000)&(df.Purchase <25000))


#check what are the categories for Product_ID, which are in test file but not in train file by applying subtract operation
diff_cat_in_train_test=test.select('Product_ID').subtract(train.select('Product_ID'))

#Return a new DataFrame containing rows in this frame but not in another frame. This is equivalent to EXCEPT in SQL. Intersection.
test_use=train_test_use.subtrcat(train_use).orderBy(col('row_id')).drop(col('row_id'))

#Group by
df.groupby('Age').agg({'Purchase': 'mean'}).show()
df.groupby('Age').count().show()

#Create sample
d1 = df.sample(False, 0.2, 42)#False means without replacement; 0.2 is the fraction to be kept, 42 is the seed

#Apply function to each row of a dataframe
df.select('column_name').map(lambda x:(x,1)).take(5) #printing (entry,1) tuple

#Sort a dataframe
df.orderBy(df.Purchase.desc()).show(5) #Purchase is the column name

#Adding new column
df.withColumn('Purchase_new', df.Purchase /2.0).select('Purchase','Purchase_new').show(5)

#Drop column; Remove a column
df.drop('column_name')

#Filter, Join, groupBy
people.filter(people.age > 30).join(department, people.deptId == department.id) \
  .groupBy(department.name, "gender").agg({"salary": "avg", "age": "max"})

#Join
people.join(department,people.deptId==department.deptId).drop(department.deptId)#Remove additional columns

people.join(department,people.deptId==department.deptId, how='inner').drop(department.deptId)#Remove additional columns

#Change name of column
new_df=df.selectExpr("id as id_1")
new_df=df.withColumnRenamed("money", "money_in_rupees")

#Drop a column from a dataframe
df_final=df_final.drop('age')

#Write a dataframe as csv
df_final.write.csv('/index/df_final.csv')

#Set this RDD’s storage level to persist its values across operations after the first time it is computed. This can only be used to assign a new storage level if the RDD does not have a storage level set yet. If no storage level is specified defaults to (MEMORY_ONLY).
rdd.persist()
df.persist()

#cache an rdd
rdd.cache()

#broadcast variables, which can be used to cache a value in memory on all nodes, and accumulators, which are variables that are only “added” to, such as counters and sums.
b = sc.broadcast([1, 2, 3, 4, 5])

#UDF with list as input
topic_words=['good','well','best']#List of words
def label_maker_topic(tokens,topic_words):
    return "answer"
topicWord=udf(lambda tkn: label_maker_topic(tkn,topic_words),StringType())#label_maker_topic is the name of the function, tkn referes to the column
myDF=myDF.withColumn("topic_word_count",topicWord(myDF.bodyText_token))#bodyText_token is the column of the dataframe

#Splitting a column into seperate columns
user_messages_sparkdf.withColumn('ads_splitted',split(col('ads'),','))

#Type Casting : changing datatype of a column in pyspark
df_num = df.select(df.employment.cast("float"), 
df.education.cast("float"), 
df.health.cast("float"))

#Filter based on length of lists in a column
df_factlist.filter(size(df_factlist['fact_list'])>=2).show()
df.where(size(col("tokens")) <= 3).show()#Another example


#Split before explode
new_final_data=final_data.withColumn('new_col_after_splitting',split(col('col_to_be_splitted'),'<seperator>'))
#Explode: Make new rows by splitting a column of list
#Given dataframe df
# +---+---------+---------+---+
# |  a|        b|        c|  d|
# +---+---------+---------+---+
# |  1|[1, 2, 3]|[7, 8, 9]|foo|
# +---+---------+---------+---+
df_exploded = df.withColumn('b', explode('b'))
# >>> df_exploded.show()
# +---+---+---------+---+
# |  a|  b|        c|  d|
# +---+---+---------+---+
# |  1|  1|[7, 8, 9]|foo|
# |  1|  2|[7, 8, 9]|foo|
# |  1|  3|[7, 8, 9]|foo|
# +---+---+---------+---+

#groupby and count and sort by count
data_use.groupby(['sal','deg']).count().orderBy('count',ascending=False).show(100)



#Window Functions
rdd = sc.parallelize([("user_1",  "object_1",  3), 
                      ("user_1",  "object_2",  2), 
                      ("user_2",  "object_1",  5), 
                      ("user_2",  "object_2",  2), 
                      ("user_2",  "object_2",  6)])
df = sqlContext.createDataFrame(rdd, ["user_id", "object_id", "score"])
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, col

window = Window.partitionBy(df['user_id']).orderBy(df['score'].desc())

df.select('*', rank().over(window).alias('rank')) 
  .filter(col('rank') <= 2) 
  .show() 
#+-------+---------+-----+----+
#|user_id|object_id|score|rank|
#+-------+---------+-----+----+
#| user_1| object_1|    3|   1|
#| user_1| object_2|    2|   2|
#| user_2| object_2|    6|   1|
#| user_2| object_1|    5|   2|
#+-------+---------+-----+----+


li=[23,34,56] #list of elements
df.filter(df['column_name'].isin(li)) #Checking if column matches any element of a list

df.filter(df['column_name'].isin(li)==False) #Selecting if column does not match any element of a list


import numpy as np
import pyspark.sql.functions as func

def median(values_list):
    med = np.median(values_list)
    return float(med)
udf_median = func.udf(median, FloatType())

df_grouped = df.groupby(['a', 'd']).agg(udf_median(func.collect_list(col('c'))).alias('median'))
df_grouped.show()

#Extract a column from a dataframe to a list
sea_lists=[row[0] for row in dataframe_with_sea.collect()]


#Round off a column
from pyspark.sql.functions import pow, lit
from pyspark.sql.types import LongType
num_places = 3
m = pow(lit(10), num_places).cast(LongType())
df = sc.parallelize([(0.6643, ), (0.6446, )]).toDF(["x"])
df.withColumn("trunc", (col("x") * m).cast(LongType()) / m)


######GROUPBY THEN CONCAT OR MAKE A LIST#####
df = spark.createDataFrame([
  ("username1", "friend1"),
  ("username1", "friend2"),
  ("username2", "friend1"),
  ("username2", "friend3")],["username", "friend"])

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
join_ = F.udf(lambda x: ", ".join(x), StringType())

#Group concat, Groupconcat, group_concat
df.groupBy("username").agg(join_(F.collect_list("friend").alias("friends_grouped"))).show(10)

#Groupby and list form , collect_list
df.groupBy("username").agg(F.collect_list("friend").alias("friends_grouped")).show(10)

def top_ss(ss_list):
	tsk = str(Counter(ss_list).most_common(50))
	return tsk


from pyspark.sql.functions import collect_list
udf_top = udf(top_ss, StringType())
final_data = useful_data.groupBy("single_col_l").agg(udf_top(collect_list(col('single_col_2'))).alias('ss_frequencies'))



#Select max or maximum from a column
train.select(max("datetime")).show(truncate=False)

#Get Item : Extract item from a specific postion of a column consisting of lists
#Previously id was [ab,fg,fe] out of which new_id [ab] is to be selected
ans=df_tmp.withColumn('new_id',split(df_tmp.id,',').getItem(0))


#Add row number column to a dataframe: Useful as pyspark dataframes cannot be accessed by index, no command like tail and join reshuffles them
df.withColumn("id", monotonically_increasing_id()).show()

#Relacing null values, missing values
train_test=train_test.na.fill({'siteid':3696590,'browserid_merged':2, 'devid_encode':1})
#siteid, browserid_merged are column names
#Sellect not null values of a column
df1.filter(df1.ColumnName_to_check.isNotNull()).show()




#UDF return multiple columns
#Source: https://stackoverflow.com/questions/35322764/apache-spark-assign-the-result-of-udf-to-multiple-dataframe-columns
from pyspark.sql.functions import udf
from pyspark.sql.types import *

schema = StructType([
    StructField("foo", FloatType(), False),
    StructField("bar", FloatType(), False)
])

def udf_test(n):
    return (n / 2, n % 2) if n and n != 0.0 else (float('nan'), float('nan'))

test_udf = udf(udf_test, schema)
df = sc.parallelize([(1, 2.0), (2, 3.0)]).toDF(["x", "y"])

foobars = df.select(test_udf("y").alias("foobar"))
foobars.printSchema()
## root
##  |-- foobar: struct (nullable = true)
##  |    |-- foo: float (nullable = false)
##  |    |-- bar: float (nullable = false)

foobars.select("foobar.foo", "foobar.bar").show()




###Case when in pyspark SOURCE: https://stackoverflow.com/questions/39982135/apache-spark-dealing-with-case-statements
from pyspark.sql import functions as F
df.select(df.name, F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()
df.withColumn('new_col', F.when(df.age > 4, 1).when(df.age < 3, -1).otherwise(0)).show()

'''
+-----+--------------------------------------------------------+
| name|CASE WHEN (age > 4) THEN 1 WHEN (age < 3) THEN -1 ELSE 0|
+-----+--------------------------------------------------------+
|Alice|                                                      -1|
|  Bob|                                                       1|
+-----+--------------------------------------------------------+
'''

from pyspark.sql import functions as F
df.select(df.name, F.when(df.age > 3, 1).otherwise(0)).show()
'''
+-----+---------------------------------+
| name|CASE WHEN (age > 3) THEN 1 ELSE 0|
+-----+---------------------------------+
|Alice|                                0|
|  Bob|                                1|
+-----+---------------------------------+
'''
##Select maximum date i.e. latest date
#Source: https://stackoverflow.com/questions/38377894/how-to-get-maxdate-from-given-set-of-data-grouped-by-some-fields-using-pyspark

from pyspark.sql.functions import col, max as max_

df = sc.parallelize([
    ("2016-04-06 16:36", 1234, 111, 1),
    ("2016-04-06 17:35", 1234, 111, 5),
]).toDF(["datetime", "userId", "memberId", "value"])

(df.withColumn("datetime", col("datetime").cast("timestamp"))
    .groupBy("userId", "memberId")
    .agg(max_("datetime")))


#After writing dataframe it is read with column _c0, _c1, _c2 ... etc. ; This script renames _c0, _c1, _c2 ... to names given in list li
li=[<colnames>]
st=''
for i in range(len(li)):
	st=st+".withColumnRenamed('_c"+str(i)+"','"+str(li[i])+"')"



#Typecasting to date format
#Source: https://stackoverflow.com/questions/38080748/convert-pyspark-string-to-date-format
from datetime import datetime
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DateType

# Creation of a dummy dataframe:
df1 = sqlContext.createDataFrame([("11/25/1991","11/24/1991","11/30/1991"), 
                            ("11/25/1391","11/24/1992","11/30/1992")], schema=['first', 'second', 'third'])

# Setting an user define function:
# This function converts the string cell into a date:
func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())

df = df1.withColumn('test', func(col('first')))


#Taking date difference in pyspark
#Source: https://stackoverflow.com/questions/36051299/how-to-subtract-a-column-of-days-from-a-column-of-dates-in-pyspark
from pyspark.sql import Column

def date_sub_(c1: Column, c2: Column) -> Column:
    return ((c1.cast("timestamp").cast("long") - 60 * 60 * 24 * c2)
        .cast("timestamp").cast("date"))

from datetime import datetime
todays_date = datetime.today().strftime("%Y-%m-%d")
def date_diff_cal(d1):
	try:
		d1_formatted = datetime.strptime(d1, '%Y-%m-%d')
		ans = (datetime.strptime(todays_date, '%Y-%m-%d') - d1_formatted).days*1.0
	except:
		ans = np.nan
	return ans

func_date_diff = udf(date_diff_cal, FloatType())
full_data = full_data.withColumn("age_calculated_in_years", func_date_diff(data.date_of_birth)/365)
full_data = full_data.withColumn("last_login_from_today", func_date_diff(full_data.last_login_date))



#Aggregate within count
import pyspark.sql.functions as func

new_log_df.cache().withColumn("timePeriod", encodeUDF(new_log_df["START_TIME"])) 
  .groupBy("timePeriod")
  .agg(
     func.mean("DOWNSTREAM_SIZE").alias("Mean"), 
     func.stddev("DOWNSTREAM_SIZE").alias("Stddev"),
     func.count(func.lit(1)).alias("Num Of Records")
   )
  .show(20, False)

#count with countDistinct
data.groupBy('col_name').agg(F.countDistinct(F.col("id")).alias("unique_ids"), func.count('col_name').alias("count_of_col")).show()

#write in s3 in parquet format
data.write.option("compression","none").save('s3://name_of_s3_bucket/folder',format="parquet",mode="overwrite")	
	
#read data from s3
df = spark.read.parquet("s3://bucket_name/folder-name")
	
#Filtering using udf : Source: https://gist.github.com/samuelsmal/feb86d4bdd9a658c122a706f26ba7e1e
from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType

def regex_filter(x):
    regexs = ['.*ALLYOURBASEBELONGTOUS.*']
    
    if x and x.strip():
        for r in regexs:
            if re.match(r, x, re.IGNORECASE):
                return True
    
    return False 
    
    
filter_udf = udf(regex_filter, BooleanType())

df_filtered = df.filter(filter_udf(df.field_to_filter_on))


####################### submitting spark applications ##########################################
#Source: https://sparkour.urizone.net/recipes/submitting-applications/
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("submitting_applications").getOrCreate()
#write the code here
...
..
spark.stop()


##FROM SHELL
$nohup /opt/spark-2.1.0-bin-hadoop2.7/bin/./spark-submit --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3 /index/job_scoring_daily/pyspark_code.py &


#Consecutive rows difference
#Link: https://www.arundhaj.com/blog/calculate-difference-with-previous-row-in-pyspark.html
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.window import Window
sc = SparkContext(appName="PrevRowDiffApp")
sqlc = SQLContext(sc)
rdd = sc.parallelize([(1, 65), (2, 66), (3, 65), (4, 68), (5, 71)])
df = sqlc.createDataFrame(rdd, ["id", "value"])
my_window = Window.partitionBy().orderBy("id")
df = df.withColumn("prev_value", F.lag(df.value).over(my_window))
df = df.withColumn("diff", F.when(F.isnull(df.value - df.prev_value), 0).otherwise(df.value - df.prev_value))
df.show()


#Modify column based on other column
#Source: https://stackoverflow.com/questions/43988801/pyspark-modify-column-values-when-another-column-value-satisfies-a-condition
from pyspark.sql.functions import when
df.withColumn('Id_New',when(df.Rank <= 5,'yes').otherwise('other')).show()

#Pandas dataframe to spark dataframe
converted_to_spark_df = spark.createDataFrame(pd_df.astype(str)).show()

#Spark dataframe to pandas dataframe
converted_to_pandas_df = spark_df.toPandas()

#pandas install in EMR so that it becomes available from pyspark3 jupyter notebook and toPandas() work
#login to EMR cluster from terminal using ssh : ssh -i abc.pem ec2-user@ec2-8757859vftsgjvbvbsv-ap-south-1.compute.amazonaws.com
$sudo python3 -m pip install pandas



final_data.select(dayofmonth(final_data.col_formatted_as_datetype)).show()

#Weekday name and encided_id
final_data.select(date_format(final_data.col_formatted_as_datetype,'u').alias('weekday_encoded_as_number'),date_format(final_data.col_formatted_as_datetype,'E').alias('weekday_as_string')).show()

##Select probability from the output of a model like randomforest
from pyspark.sql.functions import udf
from pyspark.sql.types import FloatType
secondelement=udf(lambda v:float(v[1]),FloatType())
transformed.select(secondelement('probability')) #here transformed is the obtained dataset

from pyspark.sql.types import FloatType
#Extracting only the column with probability with column with 1's probability
secondelement=udf(lambda v:float(v[1]),FloatType())
transformed.select(secondelement('probability')).show()

#Dataframe column to list
mvv_count_df.select('mvv').collect()

#-------- creating saving loading model ------------------
rf = RandomForestClassifier(labelCol='label', featuresCol='features',numTrees=20)

paramGrid = ParamGridBuilder().build()#ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01, 0.001, 0.0001]).build()
#lr = LinearRegression()
#paramGrid = ParamGridBuilder().addGrid(lr.maxIter, [500]).addGrid(lr.regParam, [0]).addGrid(lr.elasticNetParam, [1]).build()
pipeline_new = Pipeline(stages=[rf])
evaluator = MulticlassClassificationEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("f1")  #/setMetricName/ "f1" (default), "weightedPrecision", "weightedRecall", "accuracy"
#evaluator = RegressionEvaluator(metricName="mae")
crossval = CrossValidator(estimator=pipeline_new, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=10)
model_new_rf = crossval.fit(trainingData)
model_new_rf.bestModel
model_new_rf.bestModel.save('rf_pipeline_model_saved')
model_new_rf.avgMetrics

#loading a saved model
from pyspark.ml import PipelineModel
loadedModel = PipelineModel.load("rf_pipeline_model_saved")


#Checkpointing is a process of truncating RDD lineage graph and saving it to a reliable distributed (HDFS) or local file system.
sc.setCheckpointDir("hdfs://hadoop-master:9000/data/checkpoint")
df.repartition(100)


#read / write parquet files
df.write.option("compression","none").save("hdfs://address/folder",format="parquet",mode="overwrite")
spark.read.parquet("hdfs://address/folder")
df.write.option("compression","snappy").parquet("hdfs://address/folder")


#Assign unique continuous numbes to rows of a dataframe
Z = spark.createDataFrame(d.select("colid").distinct().rdd.map(lambda x: x[0]).zipWithUniqueId())


#A window function calculates a return value for every input row of a table based on a group of rows, called the Frame
from pyspark.sql.window import Window
window = Window.partitionBy(tmp['prediction']).orderBy(df['clust_count'].desc())
tmp.select('*', F.rank().over(window).alias('rank')) .filter(col('rank') <= 5).head(5)


#K-means cluster	
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import StandardScaler
d = scaler.fit(d).transform(d)
kmeans = KMeans(k=5, seed=1, featuresCol="scaled")
model = kmeans.fit(d)
centers = model.clusterCenters()


#Converting date to timestamp
x = x.withColumn("unix_time", F.unix_timestamp(F.col("DATETIME"), format='yyyy-MM-dd HH:mm:ss'))

#pivot(pivot_col, values=None); pivot_col – Name of the column to pivot. values – List of values that will be translated to columns in the output DataFrame
df4.groupBy("year").pivot("course", ["dotNET", "Java"]).sum("earnings").collect()

#Groupby and count distinct / unique
df.groupby("device").agg(F.countDistinct(F.col("colid"))).toPandas()
df.groupby("device").agg(F.countDistinct(F.col("id"))).orderBy('count(DISTINCT id)', ascending = False)

#Show Unique, Distinct records of a column
data.select("column").distinct().show(400)

#filter by date
from datetime import datetime
data = data_raw.withColumn('date_new_format',sent_data_raw.date.cast("timestamp")) #date is the column name here
data.filter(data.date_new_format>datetime.strptime('2018-04-01', '%Y-%m-%d')).show(10)

#Random Sample extract from a dataframe
df.sample(withReplacement=False, fraction=0.5, seed=None)
df.sample(False, 0.5, 42)

#If writing dataframe is giving error like stage failure
df.write.option("compression","none").save("/raw_data",format="parquet",mode="overwrite")


###############LEARNINGS / LESSONS FROM MISTAKES###############
1) While bringing data from Hadoop to Local data system, count (number of rows) match then only proceed
2)
