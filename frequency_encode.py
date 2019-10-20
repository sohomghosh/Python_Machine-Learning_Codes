#https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
#Frequency encode together; df1 is the training set and df2 is the test set
#cols is a list of categorical columns
#Refernce: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600

def encode_FE(df1,df2,cols):
    for col in cols:
        df = pd.concat([df1[col],df2[col]])
        vc = df.value_counts(dropna=True, normalize=True).to_dict()
        vc[-1] = -1
        nm = col+'_FE'
        df1[nm] = df1[col].map(vc)
        df1[nm] = df1[nm].astype('float32')
        df2[nm] = df2[col].map(vc)
        df2[nm] = df2[nm].astype('float32')
        print(nm, ', ' , end='')
