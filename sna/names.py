fp=open("dblp50000.xml",'r')
fp1=open("names.csv",'w')
fp1.write("Authors\n")
while True:
    line=fp.readline()
    if not line:
        break
    if "<author>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+"\n")