#import nltk; nltk.download("wordnet")
from nltk.corpus import wordnet as wn
set([i for ss in wn.synsets('hello') for i in ss.lemma_names()])
