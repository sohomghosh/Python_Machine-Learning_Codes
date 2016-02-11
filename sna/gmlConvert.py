
import networkx as nx
G = nx.Graph()

input1=open("C:\\Users\\Sid\\Desktop\\khusbu\\C_Programs\\Integer_Nodes.txt","r")
#input1=open("C:\\Users\\Sid\\Desktop\\4thyearproject\\New\\small_network\\football\\textfiles\\MaxSpanningTree(football)4.txt","r")
while True:
    s = input1.readline()
    if not s: break
    a = s.split()[0]
    b = s.split()[1]
    #c = s.split()[2]
    a = int(a)
    b = int(b)
    #c = float(c)
    if b!=0:
        G.add_node(a)
        G.add_node(b)
        G.add_edge(a, b)# weight = c)        
input1.close()
#nx.write_gml(G,"C:\\Users\\Sid\\Desktop\\result.gml")
nx.write_gml(G,"C:\\Users\\Sid\\Desktop\\khusbu\\C_Programs\\Integer_Nodes.gml")
