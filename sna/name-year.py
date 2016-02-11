fp=open("dblp50000.xml",'r')
fp1=open("authorName_year.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    #tk=line.split()
    if "author" in line:
        tk=line.split('>')
        tk1=tk[1].split('<')
        fp1.write(str(tk1[0])+",")
    if "year" in line:
        tk=line.split('>')
        tk1=tk[1].split('<')
        fp1.write(str(tk1[0])+"\n")
fp.close()
fp1.close()