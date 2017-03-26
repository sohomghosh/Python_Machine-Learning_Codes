import re
import urllib3
http = urllib3.PoolManager()
for line in open("list.txt",'r'):
	line1=re.sub(' ','%20',str(line))
	r=http.request('GET','http://<url_here>?s='+str(line1)[:-1]+'&maxcount=30')
	tk=str(r.data)[3:-2]
	tk_skills=re.findall(r'"\s*([^"]*?)\s*"', tk)
