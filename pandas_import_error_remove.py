import csv
import pandas as pd
data=pd.read_csv("file.csv", header = None, delimiter="\t", quoting=csv.QUOTE_NONE, encoding='utf-8')
data.head()
