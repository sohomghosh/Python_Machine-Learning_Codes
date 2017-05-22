df1=pd.DataFrame({'count' : df1.groupby( [ "Name", "City"] ).size()}).reset_index()
