#####INSTALLATIONS#########
#yum install python-flask
#pip3 install flask


#######CODE#############

#!/usr/bin/python
from flask import Flask
from flask import request

app = Flask(__name__)

@app.route('/hello')
def api_hello():
	if 'name' in request.args:
		return 'Hello ' + request.args['name']
	else:
		return 'Hello John Doe'


if __name__ == '__main__':
	app.run(host="174.89.99.186",debug=True)


#run by ./python_flask_file.py
#call by http://174.89.99.186:5000/hello?name=SohomGhosh
#call by http://174.89.99.186:5000/hello
