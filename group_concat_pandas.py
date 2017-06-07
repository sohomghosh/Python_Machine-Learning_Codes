df.groupby('team').apply(lambda x: ','.join(x.user))

OR

df.groupby('team').agg({'user' : lambda x: ', '.join(x)})
