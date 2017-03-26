import numpy as np
import pandas as pd
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt

data=pd.read_csv("file.csv")
text=data[data['Name'] == 'DDT']
text=data["comments"]
wordcloud = WordCloud().generate(str(text))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
