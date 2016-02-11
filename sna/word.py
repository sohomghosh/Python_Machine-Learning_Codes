fp=open("title.txt",'r')
fp1=open("word_freqYEAR.csv",'w')
fp1.write("Word, Year\n")
while True:
    line=fp.readline()
    if not line:
        break
    tk=line.split('@')
    tk1=tk[0].split(' ')
    for i in range(0,len(tk1)-1):
        fp1.write(str(tk1[i])+","+str(tk[1])+"\n")
fp.close()
fp1.close()