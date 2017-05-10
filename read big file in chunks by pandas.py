import pandas as pd
import numpy as np

data=pd.concat(pd.read_csv("file_name_downloaded_from_hive",sep='\x01',error_bad_lines=False,header=0,low_memory=False,chunksize=16*1024))
