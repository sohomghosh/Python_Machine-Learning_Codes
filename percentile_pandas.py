#Keep only those people whose age is between 65 and 95 percentle
data=data[(data['age']>=np.nanpercentile(data['age'],65))&(data['age']<=np.nanpercentile(data['age'],95))]
