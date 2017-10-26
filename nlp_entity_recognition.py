#pip3 install -U spacy
#python3 -m spacy.en.download all

import spacy
nlp=spacy.load('en')sentence="Ram of Apple Inc. travelled to Sydney on 5th October 2017"
for token in nlp(sentence):
   print(token, token.ent_type_)

