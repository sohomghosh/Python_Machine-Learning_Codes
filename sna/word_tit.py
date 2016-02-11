fp=open("dblp50000.xml",'r')
fp1=open("title.txt",'w')
while True:
    line=fp.readline()
    if not line:
        break
    if "<title>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+"@")
    if "<year>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+"\n")
fp.close()
fp1.close()