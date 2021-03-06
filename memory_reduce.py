#Reference: https://www.kaggle.com/sunilsj99/fraud-detection-ieee
#If file is big in size split it vertically from shell / bash commands use the following commans as a reference to split huge files vertically into chunks (Reference: http://www.unixcl.com/2009/10/awk-split-file-vertically-on-columns.html)
#The following code will create 2 files from file.txt which has 9 columns. The first file will have columns 1 to 3 and the second column will have columns 4 to 9
#$cut -d"," -f1-3 file.txt >file_col1_to_3.txt
#$cut -d"," -f4-9 file.txt >file_col4_to_9.txt
#$cut -d$"\t" -f4-9 tsv_file.tsv >tsv_file_col4_to_9.tsv ##splitting tsv files
#To rejoin and get back the original file use the following bash/shell command
#$paste -d "," file_col1_to_3.txt file_col4_to_9.txt >merged_file.txt
import pandas as pd
import numpy as np
def reduce_mem_usage(props):
    """
    'props' : is a dataframe whose memory usage is to be reduced
    Usage: import memory_reduce; df_reduced = memory_reduce.reduce_mem_usage(df);
    """
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props
