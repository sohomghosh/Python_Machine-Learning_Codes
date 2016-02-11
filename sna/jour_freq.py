fp=open("dblp50000.xml",'r')
fp1=open("author_count.txt",'w')
k=0
while True:
    line=fp.readline()
    if not line:
        break
    if "<author>" in line:
        #print "hello"
        line1=line.split('>')
        #print line1[1]
        line2=line1[1].split('<')
        #print line2[0]
        k=k+1
        #fp1.write(str(line2[0])+"@")
    if "<year>" in line:
        line1=line.split('>')
        #print line1[1]
        line2=line1[1].split('<')
        #print line2[0]
        fp1.write(str(k)+"\n")
        k=0
fp.close()
fp1.close()