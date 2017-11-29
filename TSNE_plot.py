import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
model = TSNE(n_components=2, random_state=0)
aa=model.fit_transform(X)

wrong=plt.scatter(aa[0:np.shape(X_train)[0],0],aa[0:np.shape(X_train)[0],1],color='red')
right=plt.scatter(aa[np.shape(X_train)[0]:,0],aa[np.shape(X_train)[0]:,1],color='blue')
plt.legend((wrong,right),('Wrong','Right'),loc='lower left',fontsize=8)
plt.show(block=True)
