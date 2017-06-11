#See if anything is common between a list 'similar_pos' and column 'pos_spec' of a dataframe data
similar_pos=['123','322']
#data['pos_spec']=['123|452','987|321']
dt_pos=data['pos_spec'].apply(lambda x : len(set(str(x).split('|')).intersection(set(similar_pos)))>0)
