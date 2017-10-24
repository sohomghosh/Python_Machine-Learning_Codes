import pandas as pd
import numpy as np
import gc

train=pd.concat(pd.read_csv("/index/sohom_experiment/av_ctr_sep17/train.csv",sep=',',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))
gc.collect()

del train['ConversionStatus']
gc.collect()

del train['ConversionDate']
gc.collect()

train['ClickDate']=pd.to_datetime(train['ClickDate'])
gc.collect()
train['dayofweek']=train['ClickDate'].dt.dayofweek
gc.collect()
train['hour']=train['ClickDate'].dt.hour
gc.collect()
train['minute']=train['ClickDate'].dt.minute
gc.collect()

del train['ClickDate']
gc.collect()


################### String to lower TRAIN
train['Device']=train['Device'].apply(lambda x:str(x).lower())
gc.collect()
train['RefererUrl']=train['RefererUrl'].apply(lambda x:str(x).lower())
gc.collect()
train['Browser']=train['Browser'].apply(lambda x:str(x).lower())
gc.collect()
train['OS']=train['OS'].apply(lambda x:str(x).lower())
gc.collect()

train.to_csv('train_clean',index=False)
