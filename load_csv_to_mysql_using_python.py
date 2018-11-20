#Reference : https://stackoverflow.com/questions/10154633/load-csv-data-into-mysql-in-python
import csv
import MySQLdb

mydb = MySQLdb.connect(host='localhost',
    user='root',
    passwd='',
    db='mydb')
cursor = mydb.cursor()

csv_data = csv.reader(file('students.csv'))
for row in csv_data:

    cursor.execute('INSERT INTO testcsv(names, \
          classes, mark )' \
          'VALUES("%s", "%s", "%s")', 
          row)
#close the connection to the database.
mydb.commit()
cursor.close()
print "Done"
