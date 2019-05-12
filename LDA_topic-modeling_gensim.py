from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import pyLDAvis.gensim
'''
li_good_words=[]
for line in open("good_words_new",'r'):
	li_good_words.append(str(line)[:-1])
'''
#print(li_good_words)
doc_complete=[]
for line in open("file_name.txt"):
	doc=""
	wo=line
	words = re.findall(r"[\w']+", str(wo)[:-1].strip())
	#print(words)
	for word in words:
		###if str(word).lower() in li_good_words:
		doc=doc+str(word).lower()+" "
	doc_complete.append(doc)
#print(docs)

stop = set(stopwords.words('english'))
exclude = set(string.punctuation) 
lemma = WordNetLemmatizer()
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

doc_clean = [clean(doc).split() for doc in doc_complete]


import gensim
from gensim import corpora
#index.
dictionary = corpora.Dictionary(doc_clean)
# Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]


# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(doc_term_matrix, num_topics=50, id2word = dictionary, passes=50)
##bow_vector = dictionary.doc2bow(doc_clean[1])
##print(ldamodel.get_document_topics(bow_vector, minimum_probability=None, minimum_phi_value=None, per_word_topics=False))
lns=ldamodel.print_topics(num_topics=50, num_words=1000)

fp1=open("Topic_Models_IT_v1.txt",'w')
for li in lns:
	fp1.write(str(li)+"\n")
fp1.close()

fp2=open("Document_topics_distribution_IT_v1.txt",'w')
for dd in doc_clean:
	bow_vector = dictionary.doc2bow(dd)
	fp2.write(str(ldamodel.get_document_topics(bow_vector, minimum_probability=None, minimum_phi_value=None, per_word_topics=False))+"\n")
fp2.close()

feature_matrix_lda=np.zeros(shape=(train_test.shape[0],50))#as number of topics is 50

rw=0
for dd in doc_clean:
	bow_vector = dictionary.doc2bow(dd)
	lis=ldamodel.get_document_topics(bow_vector, minimum_probability=None, minimum_phi_value=None, per_word_topics=False)
	for (a,b) in lis:
		feature_matrix_lda[rw,a]=b
	rw=rw+1


feature_lda_df=pd.DataFrame(feature_matrix_lda)


texts = doc_clean#[[word for word in document.lower().split() if word not in stop] for document in documents]

from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
 	for token in text:
 		frequency[token] = frequency[token] + 1

#texts = [[token for token in text if frequency[token] > 1]
corpus = [dictionary.doc2bow(text) for text in texts]
data=pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(data,local=False)
#pyLDAvis.enable_notebook()
###print(data)
pyLDAvis.show(data, ip='127.0.0.7', port=8888, n_retries=50, local=True, open_browser=True, http_server=None)#, **kwargs)
##Numpy extend to handle complex numbers

##Change the class NumPyEncoder of utils.py present in /home/sohom/anaconda3/lib/python3.5/site-packages/pyLDAvis into the following
'''
class NumPyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int64) or isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.float64) or isinstance(obj, np.float32):
            return float(obj)
        if np.iscomplexobj(obj):
            return abs(obj)
        return json.JSONEncoder.default(self, obj)
'''        
