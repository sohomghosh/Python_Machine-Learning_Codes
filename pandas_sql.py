#From link http://pandas.pydata.org/pandas-docs/stable/comparison_with_sql.html
import pandas as pd
import numpy as np

url = 'https://raw.github.com/pandas-dev/pandas/master/pandas/tests/data/tips.csv'
tips = pd.read_csv(url)
tips.head()

##SQL => SELECT
#SELECT total_bill, tip, smoker, time
#FROM tips
#LIMIT 5;

tips[['total_bill', 'tip', 'smoker', 'time']].head(5)



##SQL => WHERE
#SELECT *
#FROM tips
#WHERE time = 'Dinner'
#LIMIT 5;

tips[tips['time'] == 'Dinner'].head(5)
#OR
#is_dinner = tips['time'] == 'Dinner'
#tips[is_dinner].head(5)


#SELECT *
#FROM tips
#WHERE time = 'Dinner' AND tip > 5.00;

tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5.00)]


