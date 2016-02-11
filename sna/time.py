fp=open("dblp50000.xml",'r')
fp1=open("time.csv",'w')
fp1.write("Authors,Year,Publisher\n")
#fg=0
while True:
    line=fp.readline()
    if not line:
        break
    if "<author>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+";")
        #print line2[0]
    if "<year>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(","+line2[0])+"\n")
        #print line2[0]
        #fg=0
fp.close()
fp1.close()