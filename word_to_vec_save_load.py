import gensim
#model.save("word2_vec_model") . ##For saving a word2vec model for future use
model=gensim.models.Word2Vec.load("word2_vec_model")
model.wv['small engine']
