#Reference: https://www.kaggle.com/dansbecker/permutation-importance/notebook


import eli5
from eli5.sklearn import PermutationImportance

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

#The values towards the top are the most important features, and those towards the bottom matter least.
