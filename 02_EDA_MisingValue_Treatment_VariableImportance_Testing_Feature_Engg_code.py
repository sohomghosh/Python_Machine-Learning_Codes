import random
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, auc,roc_curve
import pandas_profiling 
from scipy import stats

X_train = pd.read_csv("/media/sohom/X_train.csv")
y_train = pd.read_csv("/media/sohom/y_train.csv")
X_test = pd.read_csv("/media/sohom/X_validation.csv")

#DATA# X_Train.csv
#Unique_ID,C1,C2,C3,C4,C5,C6,C7,C8,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31,N32,N33,N34,N35
#Candidate_5926,1,0,11,31,0,FALSE,0,TRUE,23.75,,2.5,,,2.595,10,0,0,2,,,14,,0,,,,,,,,27.816,1750,,,,,,,,,58,113.39,12
#Candidate_48134,1,4,2,66,2,FALSE,1,TRUE,11.05,22,3.7,16,12,3.795,19,4,72,0,9,0,5,0,0,0,1944,0.06,25856,17,0.88,1,40,10833.33333,,,,,,,,,160,262.1,17

#DATA# y_train.csv
#Unique_ID,Dependent_Variable
#Candidate_5926,1
#Candidate_48134,0


#DATA# X_validation.csv
#Unique_ID,C1,C2,C3,C4,C5,C6,C7,C8,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31,N32,N33,N34,N35
#Candidate_17537,1,4,6,41,2,FALSE,4,TRUE,19,29,2.9,10,8,2.995,15,5,77,0,4,0,1,0,0,0,2550,1.05,39,11,0.9,2,60,3033.333333,1,8,8,0,0,2551,2126.51,40,51.02,93.51,12
#Candidate_21230,1,7,22,1,1,FALSE,4,TRUE,30.58,2,3.4,0,0,3.495,13,0,0,0,1,0,1,0,0,0,0,0,0,9,0.88,0,12,2916.666667,,,,,,,,,80,171.08,18


train = X_train
train['Dependent_Variable'] = y_train['Dependent_Variable']
test = X_test

train_test = train.append(test)
train_test.describe()
train_test.dtypes

#####Variable Importances
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html


##EDA
##Reference: https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
##Reference: https://towardsdatascience.com/visualizing-your-exploratory-data-analysis-d2d6c2e3b30e
##Reference: http://www.stat.cmu.edu/~hseltman/309/Book/chapter4.pdf

#EDA Objectives:
#1) detection of mistakes in data - Missing value Analysis and Treatment
#2) checking of assumptions - Making hypothesis and checking them
#3) determining relationships among the explanatory variables, and
#4) assessing the direction and rough size of relationships between explanatory and outcome variables.
#5) preliminary selection of appropriate models

###Number/Percentage of missing value in train and test
pd.isnull(train).sum()
pd.isnull(test).sum()


######Missing value treatment; Checking distribution / plots before and after missing value treatment
#Method-1: Deletion
#Method-2: Mean/ Mode/ Median Imputation
#Method-3: Prediction Model
#Method-4: KNN Imputation
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN', strategy="mean", axis=0)
#strategy: "mean" or "median" or "most_frequent"
train['N30_missing_imputed'] = imp.fit_transform(train['N30'].values.reshape(-1,1))
imp.fit_transform(train.iloc[:,1:]) #Removing first column as it is a text variable

#Reference: https://pypi.python.org/pypi/fancyimpute/0.0.4
#pip3 install fancyimpute
#ONLY NUMERIC VALUES
from fancyimpute import NuclearNormMinimization, KNN, MICE
solver = NuclearNormMinimization(min_value=0.0, max_value=1.0, error_tolerance=0.0005)
X_filled = solver.complete(train['N30'].values.reshape(-1,1))
X_filled = solver.complete(train)
X_filled_knn = KNN(k=3).complete(train)
#https://github.com/hammerlab/fancyimpute/blob/master/fancyimpute/mice.py
X_filled_mice = MICE().complete(train.as_matrix())
X_filled_mice_df = pd.DataFrame(X_filled_mice)
X_filled_mice_df.columns = train.columns
X_filled_mice_df.index = train.index
#Other methods: SimpleFill, SoftImpute, IterativeSVD, MICE, MatrixFactorization, NuclearNormMinimization, KNN, BiScaler
#SimpleFill: uses mean or median; SoftImpute: Matrix completion; 

###Smote
#Only numeric/boolean and non_null values as input to TSNE model :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
from imblearn.over_sampling import SMOTE 
sm = SMOTE(random_state=42)
X_train_new, y_train_new = sm.fit_sample(train.dropna().iloc[:,1:44], train.dropna()['Dependent_Variable'])


#####Check if sample is representing the population: Central Limit Theorem, https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test


#Hyothesis testing, Degree of Freedom, t-statistics student t-test etc.
#References: http://www.scipy-lectures.org/packages/statistics/index.html#pairplot-scatter-matrices
#scipy.stats.ttest_1samp() tests if the population mean of data is likely to be equal to a given value (technically if observations are drawn from a Gaussian distributions of given population mean). It returns the T statistic, and the p-value

##1-sample ttest
stats.ttest_1samp(data['VIQ'], 0)
stats.ttest_1samp(train['N32'].dropna(), 0)

#2-sample ttest
female_viq = data[data['Gender'] == 'Female']['VIQ']
male_viq = data[data['Gender'] == 'Male']['VIQ']
stats.ttest_ind(female_viq, male_viq)

#paired ttest
stats.ttest_ind(data['FSIQ'], data['PIQ'])
stats.ttest_1samp(data['FSIQ'] - data['PIQ'], 0) 


#Skewness and kurtosis
#Skewness is a measure of asymmetry. Kurtosis is a more subtle measure of peakedness compared to a Gaussian distribution.
from scipy.stats import kurtosis, skew
kurtosis(train['N35'].notnull())
skew(train['N35'].notnull())

#Cross-tabulation #For generally for categorical_variables
pd.crosstab(df.col_1, [df.col_2, df.col_3], rownames=['col_1'], colnames=['col_2', 'col_3'])

#pip3 install pandas-profiling
#pandas_profiling.ProfileReport(X_train) #For IPython Notebook
profile = pandas_profiling.ProfileReport(train)
rejected_variables = profile.get_rejected_variables(threshold=0.9)
profile.to_file(outputfile="pandas_profiling_output_train.html")

##Univariate Plot: Variable-1 vs its distribution plot AND min, max, mode, standard_deviation, quantiles

train_test['N35'].std()
train_test['N35'].quantile([.25, .5, .75])
bins = np.arange(train_test['N35'].min(), train_test['N35'].max(), 5)
#plt.hist(train_test[train_test['N35'].isnull()==False]['N35'], bins = bins, alpha = 1)
plt.hist(train_test[train_test['N35'].notnull()]['N35'], bins = bins, alpha = 1)
plt.show()

#Distributional Plots
import seaborn as sns
sns.set(color_codes=True)
sns.set_palette(sns.color_palette("muted"))
sns.distplot(train_test['N35'].dropna())


##Bivariate Plot: Variable-1 vs Target variable
#Scatter Plot
sns.lmplot('N35','Dependent_Variable', fit_reg = False, data=train, size = 8)
sns.plt.show()


##Multivariate Plot: Variable-1 vs Variable-2 :: Explore relationships between variables
#Heat map to see correlation 
f, ax = plt.subplots(figsize=(121,121))
sns.heatmap(train.corr(),annot=True)
plt.show()

train.plot(x = 'N35', y = ['N34', 'N33'], subplots=True)
plt.show()

#For continuous variables N33, N34, N35; only numeric values give as input and no null :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
sns.pairplot(train.dropna(), vars=['N33', 'N34', 'N35'], kind='reg')
plt.show()

#For continuous variables N33, N34, N35 and categorical variable C6 :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
seaborn.pairplot(train.dropna(), vars=['N33', 'N34', 'N35'], kind='reg', hue='C6')  
plt.show()

scatter_matrix(train[features],diagonal='hist')
plt.show()



##Clustering of similar observations in the dataset into differentiated groupings, which by collapsing the data into a few small data points, patterns of behavior can be more easily identified
##K-means : plot
#Remember only numeric values give as input and no null :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=40).fit_predict(train.iloc[:,10:43].dropna())
plt.scatter(train.iloc[:,10:43].dropna().iloc[:, 0], train.iloc[:,10:43].dropna().iloc[:, 1], c=y_pred)
plt.show()

##Hierarchical_Clustering : Dendogram plot of features
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.metrics.pairwise import cosine_similarity
dist = 1 - cosine_similarity(train.iloc[:,10:43].dropna())

linkage_matrix = ward(dist) #define the linkage_matrix using ward clustering pre-computed distances

fig, ax = plt.subplots(figsize=(100, 100),dpi=100) # set size
#ax = dendrogram(linkage_matrix, orientation="right", labels=features);
ax = dendrogram(linkage_matrix, orientation="right")
plt.tick_params(
    axis= 'x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')

plt.tight_layout() #show plot with tight layout
plt.show()
#uncomment below to save figure
plt.savefig('ward_clusters_roles.png', dpi=100)



##Dimensionality Reduction : 
#PCA
#Correlation identification with matplotlib
# Import `PCA` from `sklearn.decomposition`
from sklearn.decomposition import PCA
# Build the model
pca = PCA(n_components=2)
# Reduce the data, output is ndarray; #Only numeric and non_null values as input to PCA :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
reduced_data = pca.fit_transform(X_train.loc[:,X_train.columns != 'Unique_ID'].dropna())
# Inspect shape of the `reduced_data`
reduced_data.shape
# print out the reduced data
print(reduced_data)
plt.scatter(reduced_data[:,0], reduced_data[:,1],cmap = 'viridis')
plt.xlabel("dimension1")
plt.ylabel("dimension2")
plt.show()


#Dimensionality Reduction T-SNE
from sklearn.manifold import TSNE
model_tsne = TSNE(n_components=2, random_state=0)
#Only numeric and non_null values as input to TSNE model :: BETTER TRY THIS AFTER MISSING VALUE IMPUTATION AND ENCODING
model_2d = model_tsne.fit_transform(X_train.loc[:,X_train.columns != 'Unique_ID'].dropna())
plt.scatter(model_2d[:,0],model_2d[:,1])
plt.legend(loc='lower left',fontsize=8)
plt.xlabel("dimension1")
plt.ylabel("dimension2")
plt.show(block=True)


########Box plot
#Source: https://www.kaggle.com/xchmiao/eda-with-python
#Reference: https://seaborn.pydata.org/generated/seaborn.boxplot.html
sns.boxplot(data=train, x = 'N34')

#Draw a vertical boxplot of a continuous variable y i.e. total_bill, grouped by a categorical variable x i.e. day:
sns.boxplot(x="day", y="total_bill", data=tips)

plt.figure(figsize=(10, 5))
plt.show()

###REMOVE OUTLIERS BASED ON THIS BOX-PLOT
#Suppose lower limit decide is 30 and upper limit decided is 80 from boxplot
train = train[(train.N34)>30 & (train.N34 <80)]

mean = np.mean(train.N34, axis=0)
sd = np.std(train.N34, axis=0)
train = train[(train.N34 > mean - 2* sd) & (train.N34 < mean + 2* sd)]


'''
########Normalize: max, min scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_test.loc[:,train_test.columns != 'Unique_ID'].dropna())  
X_train_new = scaler.transform(X_train.loc[:,X_train.columns != 'Unique_ID'].dropna())  
X_test_new = scaler.transform(X_test.loc[:,X_test.columns != 'Unique_ID'].dropna())
#Number of columns in train_test and X_test should match

########Normalize: standard scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()  
scaler.fit(train_test.loc[:,train_test.columns != 'Unique_ID'].dropna())  
X_train_new = scaler.transform(X_train.loc[:,X_train.columns != 'Unique_ID'].dropna())  
X_test_new = scaler.transform(X_test.loc[:,X_test.columns != 'Unique_ID'].dropna())
#Number of columns in train_test and X_test should match

'''

#######Bin continuous variables in groups: use cut() to cut the values for a column in bins
# Define your own bins
mybins = range(int(train["N35"].min()), int(train["N35"].max()), 10)
# Cut the data with the help of the bins
train['N35_bucket'] = pd.cut(train['N35'], bins=mybins)
# Count the number of values per bucket
train['N35_bucket'].value_counts()
#Encode train['N35_bucket'] for further use
train['N35_bucket_encoded_for_use'] = train['N35_bucket'].factorize()[0]


######Feature importances using # Import `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier
names = features
# Build the model
rfc = RandomForestClassifier()#parameters give as input
# Fit the model
XX = train.dropna()
del XX['Unique_ID']
YY = train.dropna()['Dependent_Variable']
rfc.fit(XX,YY)
# Print the results
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), rfc.feature_importances_), names), reverse=True))

# Pearson correlation
train_test.corr()
# Kendall Tau correlation
train_test.corr('kendall')
# Spearman Rank correlation
train_test.corr('spearman')

##########################################################################################################################################
###############################################Data Cleaning##############################################################################
##########################################################################################################################################

##########################################################################################################################################
#####################################Check if train and test set are showing same kind of characteristics###############################
##########################################################################################################################################
#Check their summaries - describe(), plots. This is to be done in the same way as a "sample is proper reresentation of the population" is checked

##########################################################################################################################################
###############################################Feature Engineering########################################################################
##########################################################################################################################################

####One Hot Encoding
#Source: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

def one_hot_encode_fns(col_to_encode,train):
	integer_encoded = label_encoder.fit_transform(train[col_to_encode].values).reshape(train.shape[0], 1)
	onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
	onehot_encoded_df = pd.DataFrame(onehot_encoded)
	onehot_encoded_df.index = train.index
	onehot_encoded_df.columns = [col_to_encode + "one_hot" + str(i) for i in range(onehot_encoded_df.shape[1])]
	return onehot_encoded_df


col_to_encode = "C6"
#train[col_to_encode] has values like True, False, True
onehot_df = one_hot_encode_fns(col_to_encode,train)
train = pd.concat([train, onehot_df], axis = 1)
del train[col_to_encode]

####Targeting Encoding
#Source: https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features
#Theory: http://www.saedsayad.com/encoding.htm
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


'''
#BETTER APPROACH: Distribute data among train and valid
import random
train_ids = random.sample(list(data.index),int(.8*len(data.index)))
train = data[data.index.isin(train_ids)]
valid = data[~data.index.isin(train_ids)]
'''

#Importance factors also check using xgboost
X_train_all = train_test.iloc[0:X_train.shape[0],:]
X_train_xgb = X_train_all.sample(frac=0.80, replace=False)
X_valid_xgb = pd.concat([X_train_all, X_train_xgb]).drop_duplicates(keep=False)

features=list(set(X_train.columns) - set(['Unique_ID']))

dtrain = xgb.DMatrix(X_train_xgb[features], X_train_xgb['Dependent_Variable'], missing=np.nan)
dvalid = xgb.DMatrix(X_valid_xgb[features], missing=np.nan)
dtest = xgb.DMatrix(X_test[features], missing = np.nan)

nrounds = 800
watchlist = [(dtrain, 'train')]

params = {"objective": "binary:logistic","booster": "gbtree", "nthread": 16, "silent": 1,
                "eta": 0.08, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.7,
                "min_child_weight": 1,
                "seed": 2016, "tree_method": "exact"}

bst = xgb.train(params, dtrain, num_boost_round=nrounds, evals=watchlist, verbose_eval=20)

valid_preds = bst.predict(dvalid)

fpr, tpr, thresholds = roc_curve(X_valid_xgb['Dependent_Variable'], valid_preds, pos_label=1)
auc_algo = auc(fpr, tpr)
print(auc_algo)

test_preds = bst.predict(dtest)

submit = pd.DataFrame({'Unique_ID': X_test['Unique_ID'], 'Class_1_Probability': test_preds})
i=i+1
submit[['Unique_ID','Class_1_Probability']].to_csv("XGB"+str(i)+".csv", index=False)



'''
#ENSEMBLING CODE

#sub6.csv, sub18.csv and sub36.csv append
df_all=pd.read_csv("sub6.csv")
for i in [18,36]:
	df_all=df_all.append(pd.read_csv("sub"+str(i)+".csv"))

ensembled_ans=df_all.groupby('roadId',as_index=False)['noOfLanes'].agg(lambda x: x.value_counts().index[0])
ensembled_ans.to_csv("sub39.csv",index=False)

'''


###########################################################################################################################
#Plot variable importance using SRK code
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    for i, feat in enumerate(features):
        outfile.write('{0}\t{1}\tq\n'.format(i,feat))
    outfile.close()

create_feature_map(features)
bst.dump_model('xgbmodel.txt', 'xgb.fmap', with_stats=True)
importance = bst.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1), reverse=True)
imp_df = pd.DataFrame(importance, columns=['feature','fscore'])
imp_df['fscore'] = imp_df['fscore'] / imp_df['fscore'].sum()
imp_df.to_csv("imp_feat.txt", index=False)


# create a function for labeling #
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.02*height,
                '%f' % float(height),
                ha='center', va='bottom')


#imp_df = pd.read_csv('imp_feat.txt')
labels = np.array(imp_df.feature.values)
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12,6))
rects = ax.bar(ind, np.array(imp_df.fscore.values), width=width, color='y')
ax.set_xticks(ind+((width)/2.))
ax.set_xticklabels(labels, rotation='vertical')
ax.set_ylabel("Importance score")
ax.set_title("Variable importance")
autolabel(rects)
plt.savefig('dummy_feature_imp_diagram.png',dpi=1000)
plt.show()
###########################################################################################################################
#Threshold find from auc
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html

