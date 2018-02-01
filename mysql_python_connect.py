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



#MariaDB connection
query = 'select * from table_name'
query='Insert into table_name values(%s,%s)'%(value_1,value_2)
try:
    conn= pymysql.connect(host='<ip_175.55.00.229>',user='user_id',password='password',db='database_name')
    if conn:
        print('Connected to MySQL database / MariaDB')
    cursor=conn.cursor()
    cursor.execute(query)
    conn.commit()
except Error as e:
        print('Error in Mysql Query / MariaDB / Connection')
