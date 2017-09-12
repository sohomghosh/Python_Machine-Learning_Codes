from nltk.corpus import words
from textblob import TextBlob

x='The sky is blue'
x=x.strip().lower()
stnc = TextBlob(x)
wds=stnc.words
wds=[i.singularize() for i in wds]

all_eng_words=words.words()

wds_which_is_eng=[i for i in wds if i in all_eng_words]
