from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_actual, y_predicted))

#Reference: https://stackoverflow.com/questions/17197492/root-mean-square-error-in-python
