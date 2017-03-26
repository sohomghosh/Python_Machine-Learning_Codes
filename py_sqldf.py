#pip install pysqldf
from pysqldf import SQLDF, load_meat, load_births
sqldf = SQLDF(globals())
meat = load_meat()
births = load_births()
print(sqldf.execute("SELECT * FROM meat LIMIT 10;").head())
