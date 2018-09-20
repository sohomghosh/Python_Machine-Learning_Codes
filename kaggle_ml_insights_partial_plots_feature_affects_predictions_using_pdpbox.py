#Reference: https://www.kaggle.com/dansbecker/partial-plots
#Partial dependence plots show how a feature affects predictions.
#partial dependence plots are calculated after a model has been fit. 

#How does it work?
'''
We will use the fitted model to predict our outcome (probability their player won "man of the game"). But we repeatedly alter the value for one variable to make a series of predictions. We could predict the outcome if the team had the ball only 40% of the time. We then predict with them having the ball 50% of the time. Then predict again for 60%. And so on. We trace out predicted outcomes (on the vertical axis) as we move from small values of ball possession to large values (on the horizontal axis).

In this description, we used only a single row of data. Interactions between features may cause the plot for a single row to be atypical. So, we repeat that mental experiment with multiple rows from the original dataset, and we plot the average predicted outcome on the vertical axis.
'''

tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=val_X, model_features=feature_names, feature=feature_to_plot)

pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()

#Image Address: https://www.kaggleusercontent.com/kf/5832288/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..263TpNvfJHOMmWj5Vow9sQ.yWhr5ckyTJLPSev5ZP-pCirIlxeLCov9KTKADseelwI4QBYZ3ZROxTjJ-sGpwX25Jj6QlZeaGhhJzNieeWjNGIVD2THt2YbAJrWvC3FrlGDSXasBul8vlc3jbLAnzkr17gj3nQNdZUncuGReR4Ua3A.yYqQIJ8nv9V2cMOp6Ps6zw/__results___files/__results___6_0.png

#Interpreting the plot (above image)
#The y axis is interpreted as change in the prediction from what it would be predicted at the baseline or leftmost value.
#A blue shaded area indicates level of confidence

#One variable 'A' having steeper slope than other variable 'B' in each of their partial plots does not gurantee that variable 'A' has more permutation importance than variable 'B'. This is because variable 'A' could have a big effect in the cases where it varies, but could have a single value 99% of the time. In that case, permuting feat_a wouldn't matter much, since most values would be unchanged.

#2D Partial Dependence Plots
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot

features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter1  =  pdp.pdp_interact(model=tree_model, dataset=val_X, model_features=feature_names, features=features_to_plot)
pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()

#Image Address : https://www.kaggleusercontent.com/kf/5832288/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..263TpNvfJHOMmWj5Vow9sQ.yWhr5ckyTJLPSev5ZP-pCirIlxeLCov9KTKADseelwI4QBYZ3ZROxTjJ-sGpwX25Jj6QlZeaGhhJzNieeWjNGIVD2THt2YbAJrWvC3FrlGDSXasBul8vlc3jbLAnzkr17gj3nQNdZUncuGReR4Ua3A.yYqQIJ8nv9V2cMOp6Ps6zw/__results___files/__results___12_0.png
#The above plot shows predictions for any combination of Goals Scored and Distance covered.
#More yellowish means higher chances of prediction
#More bluish means lesser chances of prediction

