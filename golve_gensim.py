#download glove from http://nlp.stanford.edu/data/glove.6B.zip , unzip it and place it in /data/clickbait_detect

from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = '/data/click_bait_detect/glove.6B.50d.txt'
word2vec_output_file = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
glove2word2vec(glove_input_file, word2vec_output_file)

from gensim.models import KeyedVectors
filename = '/data/click_bait_detect/glove.6B.50d.txt.word2vec'
model = KeyedVectors.load_word2vec_format(filename, binary=False)

model['hi] #model['word_to_serach']
#OUTPUT : 50 dimensional vectors of word 'hi'


#Reference: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
