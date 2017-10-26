#python3
import pymysql
pymysql.install_as_MySQLdb()
import MySQLdb

db = MySQLdb.connect(host="172.xx.yy.zzz",    # your host, usually localhost
                     user="username",         # your username
                     passwd="password",  # your password
                     db="name_of_database")        # name of the data base

cur = db.cursor()
cur.execute("SELECT * FROM table_name")
for row in cur.fetchall():
	print(row[0])
 

db.close()
