###### USE MULTI-THREADING

import pandas as pd
import numpy as np
import gc

train=pd.concat(pd.read_csv("/index/sohom_experiment/av_ctr_sep17/train.csv",sep=',',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))
gc.collect()


##train['ConversionStatus'].value_counts()
#False    63332693
#True        34524


##train[train['ConversionStatus']==False].head()

##train[train['ConversionStatus']==False]['ConversionPayOut'].value_counts()
#0.0    63332693

##pd.isnull(train).sum() > 0
'''
ID                      False
Country                  True
Carrier                 False
TrafficType              True
ClickDate               False
Device                   True
Browser                  True
OS                       True
RefererUrl               True
UserIp                  False
ConversionStatus        False
ConversionDate           True
ConversionPayOut         True
publisherId              True
subPublisherId           True
advertiserCampaignId     True
Fraud                   False
dtype: bool
'''

##[(f,train[f].dtype) for f in train.columns]
#[('ID', dtype('int64')), ('Country', dtype('O')), ('Carrier', dtype('float64')), ('TrafficType', dtype('O')), ('ClickDate', dtype('O')), ('Device', dtype('O')), ('Browser', dtype('O')), ('OS', dtype('O')), ('RefererUrl', dtype('O')), ('UserIp', dtype('O')), ('ConversionStatus', dtype('bool')), ('ConversionDate', dtype('O')), ('ConversionPayOut', dtype('float64')), ('publisherId', dtype('O')), ('subPublisherId', dtype('O')), ('advertiserCampaignId', dtype('float64')), ('Fraud', dtype('float64'))]

#dtype('O') means object


##train['Fraud'].value_counts()
#0.0    63366821
#1.0         396


##train.groupby(['Fraud','ConversionStatus']).size()
# Fraud  ConversionStatus
# 0.0    False               63332297
#        True                   34524
# 1.0    False                    396

###Rule if fraud then ConversionPayout = 0


##set(list(train['Country']))
##PTR: There are values like  nan, '**'

##set(list(train['TrafficType']))
#{nan, 'M', 'A'}


##set(list(train['Device']))

##len(set(list(train['Device'])))
##len(set([str(i).lower() for i in list(train['Device'])]))
#These two lengths are different, so need to convert Device into lowercase

del train['ConversionStatus']
gc.collect()

del train['ConversionDate']
gc.collect()

################### Make features from ClickDate TRAIN
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




<after>
########### Make features from UserIp TRAIN
'''
train['UserIp0']=train[train['UserIp'].apply(lambda x:len(str(x).split('.'))==4]['UserIp'].apply(lambda x:str(x).split('.')[0])
train['UserIp0']=train['UserIp'].apply(lambda x:str(x).split('.')[0])
gc.collect()
train['UserIp2']=train['UserIp'].apply(lambda x:str(x).split('.')[1])
gc.collect()
train['UserIp3']=train['UserIp'].apply(lambda x:str(x).split('.')[2])
gc.collect()
train['UserIp4']=train['UserIp'].apply(lambda x:str(x).split('.')[3])
gc.collect()

del train['UserIp']
gc.collect()
'''
</after>
train.to_csv('train_clean',index=False)




#Don't load test set now, will see to it afterwards; Loading train & test set at a time may cause trouble
test=pd.concat(pd.read_csv("/index/sohom_experiment/av_ctr_sep17/test.csv",sep=',',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))
gc.collect()

################### Make features from ClickDate TEST
test['ClickDate']=pd.to_datetime(test['ClickDate'])
gc.collect()
test['dayofweek']=test['ClickDate'].dt.dayofweek
gc.collect()
test['hour']=test['ClickDate'].dt.hour
gc.collect()
test['minute']=test['ClickDate'].dt.minute
gc.collect()

del train['ClickDate']
gc.collect()


################### String to lower TEST
test['Device']=test['Device'].apply(lambda x:str(x).lower())
gc.collect()
test['refererUrl']=test['refererUrl'].apply(lambda x:str(x).lower())
gc.collect()
test['Browser']=test['Browser'].apply(lambda x:str(x).lower())
gc.collect()
test['OS']=test['OS'].apply(lambda x:str(x).lower())
gc.collect()



<after>
########### Make features from UserIp test
'''
test['UserIp0']=test['UserIp'].apply(lambda x:str(x).split('.')[0])
gc.collect()
test['UserIp2']=test['UserIp'].apply(lambda x:str(x).split('.')[1])
gc.collect()
test['UserIp3']=test['UserIp'].apply(lambda x:str(x).split('.')[2])
gc.collect()
test['UserIp4']=test['UserIp'].apply(lambda x:str(x).split('.')[3])
gc.collect()
del test['UserIp']
gc.collect()
'''
</after>

test.to_csv("test_clean",index=False)

