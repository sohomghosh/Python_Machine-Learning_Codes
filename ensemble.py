########Coorelation
#First check if the predicted values are correlated

###########Majority Voting
#For 0 or 1 predictions
#Ensembled multiple dataframes
df_all=pd.read_csv("sub6.csv")
for i in [18,36]:
	df_all=df_all.append(pd.read_csv("sub"+str(i)+".csv"))


ensembled_ans=df_all.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("sub39.csv",index=False)
#For probability btween 0 and 1 predictions


############Weighted Majority Voting
https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/kaggle_vote.py

#Motivation# Count the vote by the best model 3 times. The other 4 models count for one vote each.
#The reasoning is as follows: The only way for the inferior models to overrule the best model (expert) is for them to collectively (and confidently) agree on an alternative.

#Assign weights proportional to their accuracy;
#Assign weights inversely proportional to their error of the models;

##Refer# https://mlwave.com/kaggle-ensembling-guide/


############Simple Averaging
https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/kaggle_vote.py
Averaging works well for a wide range of problems (both classification and regression) and metrics (AUC, squared error or logaritmic loss).


############Rank Averaging
https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/src/kaggle_rankavg.py
Rank the predicted probabilities obtained for each models. Take average of these  ranks for each rows. Final prediction is obtained by normalizing the averag ranks.
THAT IS normalizing the averaged ranks between 0 and 1 you are sure to get an even distribution in your predictions.
Ranking averages do well on ranking and threshold-based metrics (like AUC) and search-engine quality metrics (like average precision at k).

Using model-1
id,prediction,rank
23,    0.7   ,2
24,    0.1   ,1
25,    0.9   ,3

Using model-2
id,prediction,rank
23,    0.1   ,1
24,    0.8   ,3
25,    0.2   ,2

#Ensembled model   #avg_rank = (rank_from_model1 + rank_from_model2)/number_of_models     ;;;;; normalized_rank = avg_rank/(number_of_rows)
id,avg_rank,normalized_rank
23, 1.5,     .5
24, 2,       .66
25, 2.5,     .83


#########Stacking

Let’s say you want to do 2-fold stacking:

Split the train set in 2 parts: train_a and train_b
Fit a first-stage model on train_a and create predictions for train_b
Fit the same model on train_b and create predictions for train_a
Finally fit the model on the entire train set and create predictions for the test set.
Now train a second-stage stacker model on the probabilities from the first-stage model(s).
A stacker model gets more information on the problem space by using the first-stage predictions as features, than if it was trained in isolation.


########Blending
In one of the 5 folds, train the models, then use the results of the models as 'variables' in logistic regression over the validation data of that fold
With blending, instead of creating out-of-fold predictions for the train set, you create a small holdout set of say 10% of the train set. The stacker model then trains on this holdout set only.
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py

When creating predictions for the test set, you can do that in one go, or take an average of the out-of-fold predictors. 
Popular non-linear algorithms for stacking are GBM, KNN, NN, RF and ET

