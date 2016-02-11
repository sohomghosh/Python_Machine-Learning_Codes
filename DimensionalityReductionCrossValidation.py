#PRINCIPAL COMPONENT ANALYSIS
import numpy as np
import matplotlib.pyplot as plt
import mlpy
np.random.seed(0)
mean, cov, n = [0, 0], [[1,1],[1,1.5]], 100
x = np.random.multivariate_normal(mean, cov, n)
pca.learn(x) #NOT WORKING
coeff = pca.coeff()
fig = plt.figure(1) # plot
plot1 = plt.plot(x[:, 0], x[:, 1], 'o')
plot2 = plt.plot([0,coeff[0, 0]], [0, coeff[1, 0]], linewidth=4, color='r') # first PC
plot3 = plt.plot([0,coeff[0, 1]], [0, coeff[1, 1]], linewidth=4, color='g') # second PC
xx = plt.xlim(-4, 4)
yy = plt.ylim(-4, 4)
plt.show()
z = pca.transform(x, k=1) # transform x using the first PC
xnew = pca.transform_inv(z) # transform data back to its original space
fig2 = plt.figure(2) # plot
plot1 = plt.plot(xnew[:, 0], xnew[:, 1], 'o')
xx = plt.xlim(-4, 4)
yy = plt.ylim(-4, 4)
plt.show()

#for other dimensionality reduction techniques USE http://mlpy.sourceforge.net/docs/3.5/dim_red.html#linear-discriminant-analysis-lda
#For cross valodation USE http://mlpy.sourceforge.net/docs/3.5/crossval.html#leave-one-out-and-k-fold
