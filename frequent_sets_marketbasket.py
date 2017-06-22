from pyspark.mllib.fpm import FPGrowth

data = sc.textFile("patterns.csv") #comma seperated items e.g. 
#apple,orange, pizza
#pizza, burger
transactions = data.map(lambda line: list(set(line.strip().split(',')))).cache()

model = FPGrowth.train(transactions, minSupport=0.001, numPartitions=10)
result = model.freqItemsets().collect()
for fi in result:
    print(fi)

#result.saveAsTextFile('frequentsets.txt')
#RUN BY
#./spark-submit --master yarn --driver-memory 3g --executor-memory 6g --executor-cores 2 --num-executors 3 <python_code_name.py> >frequentset_of_items.txt
