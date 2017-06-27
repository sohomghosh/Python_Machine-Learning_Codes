#Source: https://www.analyticsvidhya.com/blog/2016/10/spark-dataframe-and-operations/

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

#Drop column
df.drop('column_name')

#Filter, Join, groupBy
people.filter(people.age > 30).join(department, people.deptId == department.id) \
  .groupBy(department.name, "gender").agg({"salary": "avg", "age": "max"})

#ML - machine learning libraries
#Source: http://spark.apache.org/docs/latest/api/python/pyspark.html
