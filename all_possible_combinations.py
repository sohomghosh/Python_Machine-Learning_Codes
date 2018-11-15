#Combination of a given length
import itertools
li = ['23', '97', '26', '27'] #list of which combination is to be found
r = 3 #length of combinations
list(itertools.combinations(li,r))

#OUTPUT
[('23', '97', '26'), ('23', '97', '27'), ('23', '26', '27'), ('97', '26', '27')]



#All possible combinations except null combination. To include null combination replace 1 by 0 in range i.e. change range(1,len(li)+1) to range(0,len(li)+1)
import itertools
li = ['23', '97', '26', '27']
li_of_li = []
for k in range(1,len(li)+1):
    for j in [list(i) for i in list(itertools.combinations(li,k))]:
        li_of_li.append(j)

print(li_of_li)

#OUTPUT
[['23'], ['97'], ['26'], ['27'], ['23', '97'], ['23', '26'], ['23', '27'], ['97', '26'], ['97', '27'], ['26', '27'], ['23', '97', '26'], ['23', '97', '27'], ['23', '26', '27'], ['97', '26', '27'], ['23', '97', '26', '27']]
