fp=open("op.csv",'r')
fp1=open("op_binary.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    tk=line.split(',')
    if len(tk)>=3:
        for i in range(0,len(tk)-2):
            for j in range(i+1,len(tk)-1):
                fp1.write(str(tk[i])+","+str(tk[j])+","+str(tk[len(tk)-1]))
    else:
        fp1.write(str(tk[0]+",@@@SINGLEAUTHOR@@@,"+tk[1]))
fp.close()
fp1.close()