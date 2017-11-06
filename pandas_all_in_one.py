#Pandas basics
import pandas as pd
data=pd.read_csv('<url:>/<location><filename.csv>')
data.head()
data.tail()
data.shape()


#Read bad lines
data_natun=pd.read_csv("file.csv",sep='\x01', dtype={'id': np.int32, 'name': object, 'projct_id':np.int32,'address':object,'age':np.float64}, error_bad_lines=False,index_col=None,low_memory=False)

#Read data giving _csv.Error: line contains NULL byte
from io import StringIO
data=pd.concat(pd.read_csv(StringIO(''.join(l.replace('\x00', '') for l in open("data.csv"))),sep=',',error_bad_lines=False,header=0,engine='python',chunksize=16*1024,quoting=csv.QUOTE_NONE, encoding='utf-8'))


#Join no result:
If after joining two pandas dataframes you are getting no results then check the datatype of the parameter used for joining the two dataframes


#Column rearrange
final_data = final_data[['col4','col2','col1']]


#Combine 2 dataframes side by side
new_df=pd.concat([df1,df2],axis=1)
#It is important to mention axis=1 else nan values may come


#Drop dupicates / duplicate rows
data=data.drop_duplicates()


#Flat Map
[item for sublist in main_list for item in sublist]
flatten = lambda main_list: [item for sublist in main_list for item in sublist]


#Group by then join
#Every man should have min five distinct cars
c=pd.DataFrame({'is_car_cnt_more5':cleaned_data.groupby(['man_id'])['car_id'].nunique()>=5}).reset_index()
c=c[c['is_car_cnt_more5']==True]
del c['is_car_cnt_more5']
cleaned_data=pd.merge(cleaned_data,c,on=['man_id'])
cleaned_data=cleaned_data.dropna()


#Group by with multiple list unmap
#For a given animal find frequency of all the attributes
#DATA
#class_id animal_id    attributes
#1        21           walk, talk, sleep
#1        22           eat, sleep
#1        21           cry,laugh, sleep
#1        21           dance, cook, walk
cluster_attribites=pd.DataFrame({'attribute_frequency' : data_use.groupby('animal_id').apply(lambda x: str(Counter([item for sublist in [i.split(',') for i in list(x['attributes'])] for item in sublist])))}).reset_index()


#Group Concat
df.groupby('team').apply(lambda x: ','.join(x.user))

OR

df.groupby('team').agg({'user' : lambda x: ', '.join(x)})


#Read hive exported files in pandas
The default separator is "^A". In python language, it is "\x01"
import pandas as pd
data=pd.DataFrame.from_csv("000000_0",sep='\x01',index_col=None)
data.columns=['colname1', 'colname2', 'colname3', 'colname4', 'colname5', 'colname6', 'colname7', 'colname8', 'colname9']
data.head


#Index of elements after sorting
s = [2, 3, 1, 4, 5]
print(sorted(range(len(s)), key=lambda k: s[k]))
#Output: [2, 0, 1, 3, 4] #Means after sorting "1" comes in front whose postion is 2nd in the original list, "2" comes second whose postion is 0th in the original list and so on


#Read files with multi-char seperator
import pandas as pd
import numpy as np
from io import StringIO

data=pd.read_csv(StringIO(''.join(l.replace('||', '$') for l in open("data.csv"))),sep='$')


#Order by
final_data=final_data.sort_values(['person_id','role_id'])


#Column transformation
#Converting to months
final_data['company_rating']=[int(i) for i in final_data['company_rating']/30]


#Count distinct
In [2]: table
Out[2]: 
   CLIENTCODE  YEARMONTH
0           1     201301
1           1     201301
2           2     201301
3           1     201302
4           2     201302
5           2     201302
6           3     201302

In [3]: table.groupby('YEARMONTH').CLIENTCODE.nunique()
Out[3]: 
YEARMONTH
201301       2
201302       3


#################################DATA FRAME OPERATIONS##############################################################
#Creating empty dataframe
df = pd.DataFrame({c: np.repeat(0, [nrow]) for c in data['PAGENO'].unique()})

#Populating the dataframe
for row in data.iterrows():
	if list(list(row)[1])[1] in data['PAGENO'].unique():
		df.set_value(index=di[list(list(row)[1])[0]], col=list(list(row)[1])[1], value=list(list(row)[1])[2])

#Extracting one by one rows
for i in range(0,nrow):
	row_df=list(df.iloc[i,0:ncol])#iloc for extracting by index, loc for extracting by names

#Extracting rows by names
df.loc[['row_name_1', 'row_name_2'],:]

#Extracting rows by index
df.iloc[[0, 1],:]

#Extracting columns by names
df.loc[:,['column_name_1', 'column_name_2']]

#Extracting columns by index
df.iloc[:,[0, 1]]

#Setting values by names
df.set_value('row_name','column_name',value=10,takeable=False)

#Setting values by index
df.set_value(0,1,value=10,takeable=True)#Referring to the 0th row and 1st column

#Select as in sql
selected_data1_erp=data[data['Profession'] == 'Data Analyst']
selected_data2_erp=selected_data1_erp.groupby(['WorkExp'],as_index = False)['Salary'].mean()

#################################################################################################################

#Sort a dataframe by one column after the other
final_data=final_data.sort_values(['org_id','class_id','subclass_id_consider'])


#Drop na
final_data=final_data.dropna()


#Filter by list
In [5]: df = DataFrame({'A' : [5,6,3,4], 'B' : [1,2,3, 5]})

In [6]: df
Out[6]:
   A  B
0  5  1
1  6  2
2  3  3
3  4  5

In [7]: df[df['A'].isin([3, 6])]
Out[7]:
   A  B
1  6  2
2  3  3


#Import error remove
import csv
import pandas as pd
data=pd.read_csv("file.csv", header = None, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
data.head()


#######################################PANDAS SQL############################################################
#From link http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
import pandas as pd
import numpy as np

url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
tips.head()

##SQL => SELECT
#SELECT total_bill, tip, smoker, time
#FROM tips
#LIMIT 5;

tips[['total_bill', 'tip', 'smoker', 'time']].head(5)



##SQL => WHERE
#SELECT *
#FROM tips
#WHERE time = 'Dinner'
#LIMIT 5;

tips[tips['time'] == 'Dinner'].head(5)
#OR
#is_dinner = tips['time'] == 'Dinner'
#tips[is_dinner].head(5)


#SELECT *
#FROM tips
#WHERE time = 'Dinner' AND tip > 5.00;

tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5.00)]
#############################################################################################################


#Percentile calculation
#Keep only those people whose age is between 65 and 95 percentle
import numpy as np
data=data[(data['age']>=np.nanpercentile(data['age'],65))&(data['age']<=np.nanpercentile(data['age'],95))]


#Read big file in chunks
data=pd.concat(pd.read_csv("file_name_downloaded_from_hive",sep='\x01',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))


#Store count after groupby
df1=pd.DataFrame({'count' : df1.groupby( [ "Name", "City"] ).size()}).reset_index()


#Convert string column to numeric
pre_final_data['appln_id']=pd.to_numeric(pd.Series(pre_final_data['appln_id']),errors='coerce')


#Transform column by comparing with a list (intersection)
#See if anything is common between a list 'similar_pos' and column 'pos_spec' of a dataframe data
similar_pos=['123','322']
#data['pos_spec']=['123|452','987|321']
dt_pos=data['pos_spec'].apply(lambda x : len(set(str(x).split('|')).intersection(set(similar_pos)))>0)

#Combine lists of different length to a dataframe; First need to convert lists to a dataframe
#Combine dataframes of different number of rows to a single dataframe
pd.concat([df,df1], ignore_index=True, axis=1) #df and df2 are dataframes created from lists of different length


#Apply on multiple pandas columns
def quater(mn,yr):
	if mn in [1,2,3]:
		return "Q1-"+str(yr)
	elif mn in [4,5,6]:
		return "Q2-"+str(yr)
	elif mn in [7,8,9]:
		return "Q3-"+str(yr)
	elif mn in [10,11,12]:
		return "Q4-"+str(yr)
	else:
		return np.nan()

#Transforming multiple columns on df_all
df_all['quater']=df_all[['month','year']].apply(lambda x: quater(*x), axis=1)


###DO NOT USE THIS### PRODUCES ERRONEOUS RESULTS####
pd.DatetimeIndex(data['Given Date'],ambiguous ='NaT').month#######DO NOT USE THIS### PRODUCES ERRONEOUS RESULTS####
###DO NOT USE THIS### PRODUCES ERRONEOUS RESULTS####

### Find out columns having null  ###
pd.isnull(df).sum() > 0

### Find out number of null values in each column
pd.isnull(df).sum()

### Pivot ### [Row to Column]
df_new=df.pivot(index='id', columns='column_to_be_transformed_to_multiple_columns', values='column_whose_values_are_to_be_shown_in_each_cell').reset_index()

### Pivot table ### [Row to Column]
df_new=df.pivot_table(index=['id1','id2'], columns=['1st_column_to_be_transformed_to_multiple_columns','2nd_column_to_be_transformed_to_multiple_columns'], values=['1st_column_whose_values_are_to_be_shown_in_each_cell','2nd_column_whose_values_are_to_be_shown_in_each_cell']).reset_index()

### Fatten-a-hierarchical-index-in-columns after using pivot_table ###
[' '.join(col).strip() for col in df.columns.values]

### See more text each column in pandas
 pd.options.display.max_colwidth = 100
	

### Source: https://medium.com/towards-data-science/pandas-tips-and-tricks-33bcc8a40bb9
#Name: "Sohom Ghosh" to "Sohom" "Ghosh" each in seperate column; Spliting a string into multiple strings by space
df[‘name’] = df.name.str.split(" ", expand=True)

### Row to Column ###
#Different types entries mentioned in "activity" column as seperate columns. The entries of these cells will be number of occurrences
df.groupby('name')['activity'].value_counts().unstack().fillna(0)

#Time Difference between timestamps of activities of a person
df = df.sort_values(by=['name','timestamp'])
df['time_diff'] = df.groupby('name')['timestamp'].diff()

#Move each row one up in the dataframe
df[‘new_shifted_column’] = df.time_diff.shift(-1)
# -1 means one row up

#Convert Time Difference to total seconds betweeen them
df['time_diff'] = df.time_diff.dt.total_seconds()

#cumuative sum
df['money_spent_so_far'] = df.groupby(‘name’)['money_spent'].cumsum()

#Cumulative count
df2 = df[df.groupby(‘name’).cumcount()==1]


#See dtype of all columns
[(f,train[f].dtype) for f in train.columns]

#Convert all categorical features to numeric
for f in features:
    if df_final_train[f].dtype=='object':
    	lbl = LabelEncoder()
    	lbl.fit(list(df_final_train[f].values)+list(df_final_test[f].values))
    	df_final_train[f] = lbl.transform(list(df_final_train[f].values))
    	df_final_test[f]= lbl.transform(list(df_final_test[f].values))

# Equivalent to summary in python #Gives Max, Min, count, detailed description
train_clean.describe()

# Spelling correction in pandas dataframe
data['col']=data['col'].replace(to_replace=['wrong_spelling_1','wrong_spelling_2'],value='correct_spelling')

#Apply user defined function taking multiple columns of a pandas dataframe input simultaneously
df.apply(lambda row: my_test(row['a'], row['c']), axis=1)
#Source: https://stackoverflow.com/questions/16353729/pandas-how-to-use-apply-function-to-multiple-columns

#Pandas - R dataframe comparison link
http://pandas.pydata.org/pandas-docs/stable/comparison_with_r.html

#Pandas - categorical features
http://pandas.pydata.org/pandas-docs/stable/categorical.html#categorical
http://pandas.pydata.org/pandas-docs/stable/api.html#api-categorical

#Reading excel file
data6 = pd.read_excel('mar_data.xls',header=0,sheetname=1)
