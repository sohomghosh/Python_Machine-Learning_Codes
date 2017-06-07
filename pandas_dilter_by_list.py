In [5]: df = DataFrame({'A' : [5,6,3,4], 'B' : [1,2,3, 5]})

In [6]: df
Out[6]:
   A  B
0  5  1
1  6  2
2  3  3
3  4  5

In [7]: df[df['A'].isin([3, 6])]
Out[7]:
   A  B
1  6  2
2  3  3
