fp1=open("autorName_year.csv",'r')
fp2=open("eachAuthor-year.csv",'w')
while True:
    line=fp1.readline()
    if not line:
        break
    tk=line.split(',')
    for i in range(0,len(tk)-1):
        fp2.write(str(tk[i])+"@"+str(tk[len(tk)-1]))
fp1.close()
fp2.close()