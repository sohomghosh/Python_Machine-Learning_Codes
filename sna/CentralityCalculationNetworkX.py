import networkx as nx
import json
G=nx.Graph()
fp=open("names_replaced_by_nos.csv",'r')
while True:
    line=fp.readline()
    if not line:
        break
    tk=line.split(',')
    G.add_edge(int(tk[0]),int(tk[1]),weight=int(tk[2]))
#dg=nx.degree_centrality(G)
#cc=nx.closeness_centrality(G, normalized=True)
bc=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
#ec=nx.edge_betweenness_centrality(G, normalized=True, weight=None)
#eg=nx.eigenvector_centrality_numpy(G)
#json.dump(dg,open("degree_centrality.txt",'w'))
#json.dump(cc,open("closeness.txt",'w'))
json.dump(bc,open("betweeness.txt",'w'))
#json.dump(ec,open("edge_betweeness.txt",'w'))
#json.dump(eg,open("eigenvector.txt",'w'))
fp.close()