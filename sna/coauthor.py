fp=open("dblp50000.xml",'r')
fp1=open("coauthors.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    if "<author>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+",")
    if "<year>" in line:
        fp1.write("\n")