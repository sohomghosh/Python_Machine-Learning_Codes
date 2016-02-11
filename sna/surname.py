fp=open("unique_names.csv",'r')
fp1=open("sur.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    k=line.split()
    fp1.write(str(k[len(k)-1])+"\n")
fp.close()
fp1.close()