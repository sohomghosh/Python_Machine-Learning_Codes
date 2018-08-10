#REFERENCES
#https://dask.pydata.org/en/latest/cheatsheet.html
#http://dask.pydata.org/en/latest/setup/single-distributed.html
#https://dask.pydata.org/en/latest/scheduling.html
#https://dask.pydata.org/en/latest/use-cases.html
#https://dask.pydata.org/en/latest/dataframe.html
#https://dask-ml.readthedocs.io/en/latest/modules/api.html
#https://dask.pydata.org/en/latest/futures.html
#https://dask.pydata.org/en/latest/bag-api.html
#dask.pydata.org
#distributed.readthedocs.org
#https://www.analyticsvidhya.com/blog/2018/08/dask-big-datasets-machine_learning-python/

$conda install dask
$pip install dask[complete]

import dask.dataframe as dd
df = dd.read_csv('my-data.*.csv')
df = dd.read_parquet('my-data.parquet')

df['z'] = df.x + df.y
result = df.groupby(df.z).y.mean()
out = result.compute()
result.to_parquet('my-output.parquet')
df = dd.read_csv('filenames.*.csv')
df.groupby(df.timestamp.day).value.mean().compute()



import dask.bag as db
b = db.read_text('my-data.*.json')
import json
records = b.map(json.loads).filter(lambda d: d["name"] == "Alice")
records.pluck('key-name').mean().compute()
records.to_textfiles('output.*.json')
db.read_text('s3://bucket/my-data.*.json').map(json.loads).filter(lambda d: d["name"] == "Alice").to_textfiles('s3://bucket/output.*.json')


df = dd.read_parquet('s3://bucket/myfile.parquet')
b = db.read_text('hdfs:///path/to/my-data.*.json')
df = df.persist() #Persist lazy computations in memory
dask.compute(x.min(), x.max()) #multiple computations in one go


#LAZY PARALLELISM FOR CUSTOM CODE
import dask
@dask.delayed
def load(filename):
  ...
@dask.delayed
def process(data):
  ...
load = dask.delayed(load)
process = dask.delayed(process)
data = [load(fn) for fn in filenames]
results = [process(d) for d in data]
dask.compute(results)


#ASYNCHRONOUS REAL-TIME PARALLELISM
from dask.distributed import Client
client = Client()
future = client.submit(func, *args, **kwargs)
result = future.result()
for future in as_completed(futures):
  ...
L = [client.submit(read, fn) for fn in filenames]
L = [client.submit(process, future) for future in L]
future = client.submit(sum, L)
result = future.result()


#HOW TO LAUNCH DASK ON A CLUSTER
$ dask-scheduler
Scheduler started at SCHEDULER_ADDRESS:8786
host1$ dask-worker SCHEDULER_ADDRESS:8786
host2$ dask-worker SCHEDULER_ADDRESS:8786
from dask.distributed import Client
client = Client('SCHEDULER_ADDRESS:8786')


#ON A SINGLE MACHINE
client = Client()

#CLOUD DEPLOYMENT
$pip install dask-kubernetes #Google Cloud
$pip install dask-ec2 #Amazon ec2


import dask_ml.joblib
from sklearn.externals.joblib import parallel_backend
from sklearn.ensemble import RandomForestClassifier 
with parallel_backend('dask'):
    # Your normal scikit-learn code here
    model = RandomForestClassifier()

    
    
from sklearn.externals.joblib import parallel_backend
import dask_ml.joblib
from sklearn.ensemble import RandomForestRegressor

with parallel_backend('dask'):
    # Create the parameter grid based on the results of random search 
     param_grid = {
    'bootstrap': [True],
    'max_depth': [8, 9],
    'max_features': [2, 3],
    'min_samples_leaf': [4, 5],
    'min_samples_split': [8, 10],
    'n_estimators': [100, 200]
    }

# Create a based model
rf = RandomForestRegressor()

# Instantiate the grid search model
import dask_searchcv as dcv
grid_search = dcv.GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
grid_search.fit(df[[features]], df['target'])
grid_search.best_params_
