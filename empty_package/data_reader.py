import pandas as pd
def get_data(my_data_file):
	data_file=pd.read_csv(my_data_file)
	return data_file	