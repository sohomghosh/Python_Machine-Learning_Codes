import numpy as np
import pandas as pd
import collections
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

other_stopwords_to_remove = ['abracadabra', 'etc']
STOPWORDS = STOPWORDS.union(set(other_stopwords_to_remove))
stopwords = set(STOPWORDS)

data=pd.read_csv("file.csv")
text=data[data['Name'] == 'DDT']
text=data["comments"]
wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                max_words=2000,
                stopwords = stopwords, 
                min_font_size = 10).generate(str(text))

#Arguments of WordCloud
#['self', 'font_path', 'width', 'height', 'margin', 'ranks_only', 'prefer_horizontal', 'mask', 'scale', 'color_func', 'max_words', 'min_font_size', 'stopwords', 'random_state', 'background_color', 'max_font_size', 'font_step', 'mode', 'relative_scaling', 'regexp', 'collocations', 'colormap', 'normalize_plurals']


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
