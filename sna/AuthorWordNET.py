import nltk
from nltk.corpus import stopwords
stop = stopwords.words('english')
fp=open("dblp50000.xml",'r')
fp1=open("AuthorWORDNET.csv",'w')
while True:
    line=fp.readline()
    if not line:
        break
    if "author" in line:
        tk=line.split('>')
        tk1=tk[1].split('<')
        fp1.write(str(tk1[0])+"@")
    if "title" in line:
        tk=line.split('>')
        tk1=tk[1].split('<')
        a=str(tk1[0])
        a=a.lower()
        k=[i for i in a.split() if i not in stop]
        for i in range(0,len(k)):
            fp1.write(str(k[i]+","))
        fp1.write("\n")
fp.close()
fp1.close()