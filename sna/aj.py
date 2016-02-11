fp=open("dblp50000.xml",'r')
fp1=open("op.csv",'w')
#fp1.write("authors.....,TimeStamp/n")
c=0
flag=0
while True:
    line=fp.readline()
    if not line:
        break
    if "author" in line or "editor" in line:
        flag=1
        tk=line.split('>')
        tk1=tk[1].split('<')
        fp1.write(str(tk1[0])+",")
    if "year" in line and flag==1:
        tk=line.split('>')
        tk1=tk[1].split('<')
        b=str(tk1[0]).strip()
        if b>='1940' and b<='1949':
            c=1
        if b>='1950' and b<='1959':
            c=2
        if b>='1960' and b<='1969':
            c=3
        if b>='1970' and b<='1979':
            c=4
        if b>='1980' and b<='1989':
            c=5
        if b>='1990' and b<='1999':
            c=6
        if b>='2000' and b<='2003':
            c=7
        flag=0
        fp1.write(str(c)+"\n")
fp.close()
fp1.close()