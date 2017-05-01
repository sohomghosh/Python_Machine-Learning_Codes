s = [2, 3, 1, 4, 5]
print(sorted(range(len(s)), key=lambda k: s[k]))
#Output: [2, 0, 1, 3, 4] #Means after sorting "1" comes in front whose postion is 2nd in the original list, "2" comes second whose postion is 0th in the original list and so on.
