fp=open("co_author.csv",'r')
fp1=open("coauthor_new_coar.csv",'w')
fp.readline();
while True:
    line=fp.readline()
    if not line:
        break
    token=line.split(',')
    a=len(token)-2 #a is no. of authors
    if a>1:
        fp1.write(str(line))
fp.close()
fp1.close()