#https://www.kaggle.com/shelars1985/unveiling-the-power-of-superheroes-of-21st-century
#https://www.kaggle.com/benhamner/python-data-visualizations
#https://www.analyticsvidhya.com/blog/2015/05/data-visualization-python/
#http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
#https://cognitiveclass.ai/courses/data-visualization-with-python/
#https://s3.amazonaws.com/assets.datacamp.com/production/course_2493/slides/ch3_slides.pdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
import seaborn as sns
from pandas.tools.plotting import scatter_matrix

######Line Plots
var = df.groupby('BMI').Sales.sum()
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('BMI')
ax1.set_ylabel('Sum of Sales')
ax1.set_title("BMI wise Sum of Sales")
var.plot(kind='line')

#Line plots using seaborn
sns.FacetGrid(iris, hue="Species", size=6).map(sns.kdeplot, "PetalLengthCm").add_legend()

######Andrews Curves involve using attributes of samples as coefficients for Fourier series
from pandas.tools.plotting import andrews_curves
andrews_curves(data.drop("Id", axis=1), "Labels")

######Parallel coordinates plots each feature on a separate column & then draws lines connecting the features for each data sample
from pandas.tools.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")

######Radviz puts each feature as a point on a 2D plane, and then simulates having each sample attached to those points through a spring weighted by the relative value for that feature
from pandas.tools.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")


#######Stacked Column Chart
var = df.groupby(['BMI','Gender']).Sales.sum()
var.unstack().plot(kind='bar',stacked=True,  color=['red','blue'], grid=False)

#######Scatter Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales']) #You can also add more variables here to represent color and size.
plt.show()

######Sactter Plot with Regression Plot
sns.lmplot(x= 'total_bill', y='tip', data=tips)
plt.show()

#Using different colors for different sex
sns.lmplot(x='total_bill', y='tip', data=tips, hue='sex', palette='Set1')
plt.show()

#Creating seperate sub_plots
sns.lmplot(x='total_bill', y='tip', data=tips, col='sex')
plt.show()

#Residual plots: Plot the residuals of a linear regression. Residuals mean the difference between the observed value of the dependent variable (y) and the predicted value (ŷ)
sns.residplot(x='age',y='fare',data=tips,color='indianred')
plt.show()

########Strip Plot : Draw a scatterplot where one variable is categorical.
sns.stripplot(y= 'tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

#with grouping strip plot
sns.stripplot(x='day', y='tip', data=tip)
plt.ylabel('tip ($)')
plt.show()

#Spreading out strip plot
sns.stripplot(x='day', y='tip', data=tip, size=4, jitter=True)
plt.ylabel('tip ($)') 
plt.show()

##########Swarm Plot : Draw a categorical scatterplot with non-overlapping points.
sns.swarmplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.show()

#grouping with swarm plots
sns.swarmplot(x='day', y='tip', data=tips, hue='sex') 
plt.ylabel('tip ($)')
plt.show()

#changing orientation
sns.swarmplot(x='tip', y='day', data=tips, hue='sex',orient='h') 
plt.xlabel('tip ($)')
plt.show()

#########Violin Plot: Draw a combination of boxplot and kernel density estimate. kernel density estimation (KDE) is a non-parametric way to estimate the probability density function of a random variable.
plt.subplot(1,2,1)
sns.boxplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.subplot(1,2,2)
sns.violinplot(x='day', y='tip', data=tips)
plt.ylabel('tip ($)')
plt.tight_layout()
plt.show()

sns.violinplot(x='day', y='tip', data=tips, inner=None, color='lightgray')
sns.stripplot(x='day', y='tip', data=tips, size=4, jitter=True)
plt.ylabel('tip ($)')
plt.show()

##########Joint Plot
sns.jointplot(x= 'total_bill', y= 'tip', data=tips)
plt.show()

##########Joint Plot with KDE
sns.jointplot(x='total_bill', y= 'tip', data=tips, kind='kde')
plt.show()

#########Pair plot
sns.pairplot(tips)
plt.show()

#with hue
sns.pairplot(tips, hue='sex')
plt.show()

########Bubble Plot
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(df['Age'],df['Sales'], s=df['Income']) # Added third variable income as size of the bubble
plt.show()

########Histograms
fig=plt.figure() #Plots in matplotlib reside within a figure object, use plt.figure to create new figure
#Create one or more subplots using add_subplot, because you can't create blank figure
ax = fig.add_subplot(1,1,1)
#Variable
ax.hist(df['Age'],bins = 7) # Here you can play with number of bins
Labels and Tit
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('#Employee')
plt.show()

#######Bar Charts
var = df.groupby('Gender').Sales.sum() #grouped sum of sales at Gender level
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Sum of Sales')
ax1.set_title("Gender wise Sum of Sales")
var.plot(kind='bar')

########Pie Charts
var=df.groupby(['Gender']).sum().stack()
temp=var.unstack()
type(temp)
x_list = temp['Sales']
label_list = temp.index
pyplot.axis("equal") #The pie chart is oval by default. To make it a circle use pyplot.axis("equal")
#To show the percentage of each pie slice, pass an output format to the autopctparameter 
plt.pie(x_list,labels=label_list,autopct="%1.1f%%") 
plt.title("Pastafarianism expenses")
plt.show()

#######Box Plots
import matplotlib.pyplot as plt
import pandas as pd
fig=plt.figure()
ax = fig.add_subplot(1,1,1)
#Variable
ax.boxplot(df['Age'])
plt.show()

#Boxplot Using seaborn
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
plt.show()

##Boxplot Using seaborn with jitter; We'll use jitter=True so that all the points don't fall in single vertical lines above the species
x = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")

########Violin Plots
import seaborn as sns 
sns.violinplot(df['Age'], df['Gender']) #Variable Plot
sns.despine()

########Heat Map
import numpy as np
#Generate a random number, you can refer your data values also
data = np.random.rand(4,2)
rows = list('1234') #rows categories
columns = list('MF') #column categories
fig,ax=plt.subplots()
#Advance color controls
ax.pcolor(data,cmap=plt.cm.Reds,edgecolors='k')
ax.set_xticks(np.arange(0,2)+0.5)
ax.set_yticks(np.arange(0,4)+0.5)
# Here we position the tick labels for x and y axis
ax.xaxis.tick_bottom()
ax.yaxis.tick_left()
#Values against each labels
ax.set_xticklabels(columns,minor=False,fontsize=20)
ax.set_yticklabels(rows,minor=False,fontsize=20)
plt.show()

f, ax = plt.subplots(figsize=(121,121))
sns.heatmap(train.corr(),annot=True)
plt.show()

#############Waffle Charts : Squared Pie Charts
#Reference: https://github.com/ligyxy/PyWaffle
#pip3 install pywaffle
data = {'Democratic': 48, 'Republican': 46, 'Libertarian': 3}
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    colors=("#983D3D", "#232066", "#DCB732"),
    title={'label': 'Vote Percentage in 2016 US Presidential Election', 'loc': 'left'},
    labels=["{0} ({1}%)".format(k, v) for k, v in data.items()],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0}
)
fig.gca().set_facecolor('#EEEEEE')
fig.set_facecolor('#EEEEEE')
plt.show()


############Word Clouds
import numpy as np
import pandas as pd
import collections
from wordcloud import WordCloud
import matplotlib.pyplot as plt
data=pd.read_csv("file.csv")
text=data[data['Name'] == 'DDT']
text=data["comments"]
wordcloud = WordCloud().generate(str(text))
#Arguments of WordCloud
#['self', 'font_path', 'width', 'height', 'margin', 'ranks_only', 'prefer_horizontal', 'mask', 'scale', 'color_func', 'max_words', 'min_font_size', 'stopwords', 'random_state', 'background_color', 'max_font_size', 'font_step', 'mode', 'relative_scaling', 'regexp', 'collocations', 'colormap', 'normalize_plurals']
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

############Folium : for ploting on Maps
#pip3 install folium
#Do this in IPython Notebook
#Examples: http://folium.readthedocs.io/en/latest/quickstart.html
import folium
map_osm = folium.Map(location=[45.5236, -122.6750], tiles='Stamen Toner', zoom_start=13)
map_osm
map_osm.save('/tmp/map.html')

map_1 = folium.Map(location=[45.372, -121.6972],
                   zoom_start=12,
                   tiles='Stamen Terrain')
folium.Marker([45.3288, -121.6625], popup='Mt. Hood Meadows').add_to(map_1)
folium.Marker([45.3311, -121.7113], popup='Timberline Lodge').add_to(map_1)
map_1

##########Maps with Markers : https://matplotlib.org/basemap/users/examples.html

############Choropleth Maps : Maps using python
https://plot.ly/python/choropleth-maps/

###########Other plots with ploty : free & open python library
https://plot.ly/python/
