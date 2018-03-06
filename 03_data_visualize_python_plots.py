------------------------------------------------------------
######################### UNIVARIATE ANALYSIS ##############
------------------------------------------------------------

##BARPLOTS (Univariate Analysis)

#Bar charts of categorical data (Univariate Analysis)
train['Gender'].value_counts().head(100).plot.bar()

#Histogram (Univariate Analysis) - For Numeric Data
train['Interest_Rate'].plot.hist()

#sns.distplot = pandas hist
sns.distplot(train['Interest_Rate'].dropna(), bins=5, kde=False)

#bar chart sorted by x-axis
train['City_Category'].value_counts().sort_index().head(100).plot.bar()

#pandas bar plot = seaborn count plot
sns.countplot(train['Gender']

#when values are precalculated
import matplotlib.pyplot as plt
df.plot(x='col_name1', y= "dependent_variable",kind='bar')
#fig = plt.figure()
#ax = fig.add_subplot(111)
#xlabel : ax.set_xlabel('xlabel')
#ylabel : ax.set_ylabel('ylabel')
#axes title : ax.set_title('axes title')
#For more see: https://matplotlib.org/users/text_intro.html
plt.show()

-----------------------------------------------------------
##LINECHARTS (Univariate Analysis)

#Line charts of numeric data (Univariate Analysis)
train['Monthly_Income'].value_counts().sort_index().plot.line()

#pandas line charts sorted by x axis
train['EMI'].value_counts().sort_index().plot.line()

---------------------------------------------------------
##KDE (Univariate Analysis)

#Kernel Density Estimate Plot (Univariate Analysis) : does smoothing
sns.kdeplot(train['Interest_Rate'].dropna())

---------------------------------------------------------

##PIECHARTS (Univariate Analysis)
train['Source_Category'].value_counts().head(10).plot.pie()
plt.gca().set_aspect('equal')

---------------------------------------------------------

##BOXPLOT (Univariate Analysis)
sns.boxplot(data=train, x = 'N34')

------------------------------------------------------------
######################### BIVARIATE ANALYSIS ##############
------------------------------------------------------------

##SCATTER PLOT (Bivariate Analysis)
train.plot.scatter(x='Loan_Amount', y='EMI')
---------------------------------------------------------------------

##KERNEL DENSITY ESTIMATE PLOT (Bivariate Analysis)
sns.kdeplot(train[['Loan_Amount','Approved']].dropna())
---------------------------------------------------------------------

##HEXPLOT (A hexplot aggregates points in space into hexagons, and then colorize those hexagons) (Bivariate Analysis)
train.plot.hexbin(x='Loan_Amount', y='EMI', gridsize=15)

---------------------------------------------------------------------

##JOINTPLOT - combine scatter and hexplot (Bivariate Analysis)
sns.jointplot(x='Loan_Amount', y='EMI', data=train[['Loan_Amount', 'EMI']].dropna())
sns.jointplot(x='Loan_Amount', y='EMI', data=train[['Loan_Amount', 'EMI']].dropna(), kind='hex', gridsize=20)

---------------------------------------------------------------------
##STACKED PLOTS (Bivariate Analysis)
train_stats_as_per_source_category = train.groupby('Source_Category').mean()[['Loan_Amount', 'Existing_EMI', 'EMI']]
train_stats_as_per_source_category.head()
'''
					Loan_Amount 		Existing_EMI 		EMI 
Source_Category
			A          16500.000000 	70.000000 			848.000000
'''

#STACKED BAR
train_stats_as_per_source_category.plot.bar(stacked=True)

#STACKED AREA
train_stats_as_per_source_category.plot.area()
---------------------------------------------------------------------
##BOXPLOT (Bivariate Analysis)
sns.boxplot(x='Source_Category', y='Loan_Amount',data=train)

---------------------------------------------------------------------
##VIOLIN PLOT (Bivariate Analysis)
sns.violinplot(x='Source_Category', y='Loan_Amount',data=train)


------------------------------------------------------------
######################### MULTI-VARIATE ANALYSIS ##############
------------------------------------------------------------
##PAIRPLOT (Multivariate plots)
sns.pairplot(train[['Existing_EMI', 'Loan_Amount', 'Monthly_Income']].dropna())

#Facet Grid (Multivariate plots) : #A FacetGrid is an object which stores some information on how you want to break up your data visualization.
g = sns.FacetGrid(train, col="Source_Category")
g.map(sns.kdeplot, "Loan_Amount")
---------------------------------------------------------------------

##SUB PLOTS (Multivariate plots)
#fig, axarr = plt.subplots(<number_of_rows>, <number_of_columns>, figsize=(<along_x_axis>, <along_y_axis>))
fig, axarr = plt.subplots(2, 2, figsize=(12, 8))
---------------------------------------------------------------------

##SCATTER PLOTS (Multivariate plots)
sns.lmplot(x='Monthly_Income', y='Existing_EMI', hue='Source_Category', data=tra
in.dropna(), fit_reg=False)
---------------------------------------------------------------------

##HEAT MAP (Multivariate Plots)
sns.heatmap(train[['Monthly_Income', 'EMI', 'Existing_EMI']].corr(),annot=True)

