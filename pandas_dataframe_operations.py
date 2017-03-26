import pandas as pd

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
