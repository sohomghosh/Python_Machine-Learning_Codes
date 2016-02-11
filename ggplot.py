import pandas as pd
from ggplot import *

budget = pd.read_csv("http://pbpython.com/extras/mn-budget-detail-2014.csv")
budget = budget.sort('amount',ascending=False)[:10]
p = ggplot(budget, aes(x="detail",y="amount")) + \
    geom_bar(stat="bar", labels=budget["detail"].tolist()) +\
    ggtitle("MN Capital Budget - 2014") + \
    xlab("Spending Detail") +  \
    ylab("Amount") + scale_y_continuous(labels='millions') + \
    theme(axis_text_x=element_text(angle=90))
print p