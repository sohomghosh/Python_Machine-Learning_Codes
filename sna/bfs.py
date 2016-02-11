import networkx as nx
fp1=open("Cit-HepPh.txt",'r')
fp1.readline()
fp1.readline()
fp1.readline()
fp1.readline()
G = nx.DiGraph()
#i=0
while True:
	line=fp1.readline()
	if not line:
		break
	tk=line.split('\t')
	#print i
	#i=i+1
	G.add_edge(int(tk[0]),int(tk[1]))
fp2=open("dfs_outputs.txt",'w')
fp2.write(str(dict(nx.bfs_successors(G,9907233))))
fp1.close()
fp2.close()