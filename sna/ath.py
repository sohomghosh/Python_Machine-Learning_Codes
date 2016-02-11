fp=open("autorName_year.csv",'r')
fp1=open("NoOfCoauthor.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    tk=line.split(',')
    if len(tk)>1:
        fp1.write(str(tk[len(tk)-1])[:-1]+"@"+str(len(tk)-1)+"\n")
fp.close()
fp1.close()