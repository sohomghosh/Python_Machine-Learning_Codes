#Every man should have min five distinct cars
c=pd.DataFrame({'is_car_cnt_more5':cleaned_data.groupby(['man_id'])['car_id'].nunique()>=5}).reset_index()
c=c[c['is_car_cnt_more5']==True]
del c['is_car_cnt_more5']
cleaned_data=pd.merge(cleaned_data,c,on=['man_id'])
cleaned_data=cleaned_data.dropna()
