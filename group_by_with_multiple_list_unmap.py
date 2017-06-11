#For a given animal find frequency of all the attributes
#DATA
#class_id animal_id    attributes
#1        21           walk, talk, sleep
#1        22           eat, sleep
#1        21           cry,laugh, sleep
#1        21           dance, cook, walk
cluster_attribites=pd.DataFrame({'attribute_frequency' : data_use.groupby('animal_id').apply(lambda x: str(Counter([item for sublist in [i.split(',') for i in list(x['attributes'])] for item in sublist])))}).reset_index()
