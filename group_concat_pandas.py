df.groupby('team').apply(lambda x: ','.join(x.user))
