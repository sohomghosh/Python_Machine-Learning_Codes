import difflib
difflib.get_close_matches('appel', ['ape', 'apple', 'peach', 'puppy'],n=3,cutoff=0.8) #n is number of elements to be returned, cutoff is the threshold for matching
#OUTPUT: ['apple']
