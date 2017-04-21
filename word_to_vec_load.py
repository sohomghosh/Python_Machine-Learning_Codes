import gensim
model=gensim.models.Word2Vec.load("word2_vec_model")
model.wv['small engine']
