In [2]: table
Out[2]: 
   CLIENTCODE  YEARMONTH
0           1     201301
1           1     201301
2           2     201301
3           1     201302
4           2     201302
5           2     201302
6           3     201302

In [3]: table.groupby('YEARMONTH').CLIENTCODE.nunique()
Out[3]: 
YEARMONTH
201301       2
201302       3
