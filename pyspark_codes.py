#Source: https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/
#https://spark.apache.org/docs/latest/ml-features.html
#Source: http://spark.apache.org/docs/latest/api/python/pyspark.html

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

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.ml.classification import GBTClassifier


from pyspark import SparkContext
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

#Distinct number of items in a column 'Item'
df.select('Item').distinct().count()

#Pair wise frequency of categorical columns
df.crosstab('Age', 'Gender').show()

#Drop duplicates
train.select('Age','Gender').dropDuplicates().show()

#Drop na
df.dropna()
df.dropna().count()

#Replace na by zero
df.fillna(0)

#Filter - Select entries for which Purchase is > 15000
df.filter(df.Purchase > 15000)

#check what are the categories for Product_ID, which are in test file but not in train file by applying subtract operation
diff_cat_in_train_test=test.select('Product_ID').subtract(train.select('Product_ID'))

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

#Change name of column
new_df=df.selectExpr("id as id_1")
new_df=df.withColumnRenamed("money", "money_in_rupees")

#Drop a column from a dataframe
df_final=df_final.drop('age')

#Write a dataframe as csv
df_final.write.csv('/index/df_final.csv')

#UDF with list as input
topic_words=['good','well','best']#List of words
def label_maker_topic(tokens,topic_words):
    return "answer"
topicWord=udf(lambda tkn: label_maker_topic(tkn,topic_words),StringType())#label_maker_topic is the name of the function, tkn referes to the column
myDF=myDF.withColumn("topic_word_count",topicWord(myDF.bodyText_token))#bodyText_token is the column of the dataframe

#Type Casting : changing datatype of a column in pyspark
df_num = df.select(df.employment.cast("float"), 
df.education.cast("float"), 
df.health.cast("float"))

#Filter based on length of lists in a column
df_factlist.filter(size(df_factlist['fact_list'])>=2).show()
df.where(size(col("tokens")) <= 3).show()#Another example



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
