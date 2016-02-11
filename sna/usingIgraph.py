from igraph import *
import json
G=Graph()
fp=open("Cit-HepPh.txt",'r')
fp.readline()
fp.readline()
fp.readline()
fp.readline()
while True:
    line=fp.readline()
    if not line:
        break
    tk=line.split('\t')
    a=int(tk[0])
    b=int(tk[1])
    G.add_edges((str(tk[0]),str(tk[1])))
#dg=nx.degree_centrality(G)
#cc=nx.closeness_centrality(G, normalized=True)
bc=g.edge_betweenness()
#ec=nx.edge_betweenness_centrality(G, normalized=True, weight=None)
#eg=nx.eigenvector_centrality_numpy(G)
#json.dump(dg,open("degree_centrality.txt",'w'))
#json.dump(cc,open("closeness.txt",'w'))
json.dump(bc,open("betweeness.txt",'w'))
#json.dump(ec,open("edge_betweeness.txt",'w'))
#json.dump(eg,open("eigenvector.txt",'w'))
fp.close()