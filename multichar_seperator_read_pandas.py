import pandas as pd
import numpy as np
from io import StringIO

data=pd.read_csv(StringIO(''.join(l.replace('||', '$') for l in open("data.csv"))),sep='$')
