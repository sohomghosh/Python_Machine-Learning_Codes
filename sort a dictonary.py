di={'aa':2,'bb':5}
sorted_dictionary=sorted(di.items(), key=lambda x: (-x[1], x[0]))
for a,b in sorted_dictonary:
 print(str(a)+":::"+str(b))
