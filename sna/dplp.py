fp=open("dblp50000.xml",'r')
fp1=open("data.csv",'w')
fp1.write("Authors,Journal/Book/Website,Year,Publisher\n")
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
    if "<journal>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(","+str(line2[0])+",")
    if "<booktitlel>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(","+str(line2[0])+",")
        #print line2[0]
    if "<year>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+",")
        #print line2[0]
    if "<publisher>" in line:
        line1=line.split('>')
        line2=line1[1].split('<')
        fp1.write(str(line2[0])+",")
        #print line2[0]
    if "</article>" in line:
        fp1.write("\n")
    if "</phdthesis>" in line:
        fp1.write("\n")
    if "</www>" in line:
        fp1.write("\n")
    if "</inproceedings>" in line:
        fp1.write("\n")

        #fg=0
fp.close()
fp1.close()