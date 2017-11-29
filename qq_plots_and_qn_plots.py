#Quantile-normal plots
#Quantile-quantile plots :: In statistics, a Q–Q (quantile-quantile) plot is a probability plot, which is a graphical method for comparing two probability distributions by plotting their quantiles against each other
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html
from scipy import stats
measurements = np.random.normal(loc = 20, scale = 5, size=100)   
stats.probplot(measurements, dist="norm", plot=pylab)
plt.show()
