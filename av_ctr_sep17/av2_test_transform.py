import pandas as pd
import numpy as np
import gc

test=pd.concat(pd.read_csv("/index/sohom_experiment/av_ctr_sep17/test.csv",sep=',',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))
gc.collect()

test['ClickDate']=pd.to_datetime(test['ClickDate'])
gc.collect()
test['dayofweek']=test['ClickDate'].dt.dayofweek
gc.collect()
test['hour']=test['ClickDate'].dt.hour
gc.collect()
test['minute']=test['ClickDate'].dt.minute
gc.collect()

del test['ClickDate']
gc.collect()

test['Device']=test['Device'].apply(lambda x:str(x).lower())
gc.collect()
test['RefererUrl']=test['RefererUrl'].apply(lambda x:str(x).lower())
gc.collect()
test['Browser']=test['Browser'].apply(lambda x:str(x).lower())
gc.collect()
test['OS']=test['OS'].apply(lambda x:str(x).lower())
gc.collect()

test.to_csv("test_clean",index=False)

