import pandas_profiling 
#pip3 install pandas-profiling #Remember only X vaiables of training set is to be given as input in pandas profiling
#pandas_profiling.ProfileReport(X_train) #For IPython Notebook
profile = pandas_profiling.ProfileReport(X_train)
rejected_variables = profile.get_rejected_variables(threshold=0.9)
profile.to_file(output_file="pandas_profiling_output_train.html")
