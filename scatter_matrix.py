import matplotlib.pyplot as plt
train_csv=pd.read_csv("/home/sohom/Desktop/HackerEarth_bank-fears-loanliness/train_indessa.csv")
from pandas.tools.plotting import scatter_matrix
import pylab
features = list(np.setdiff1d(train_csv.columns, ['emp_title', 'desc','purpose','title','zip_code','addr_state','loan_status']))
train_csv[features]
scatter_matrix(train_csv[features],diagonal='hist')
pylab.show()
