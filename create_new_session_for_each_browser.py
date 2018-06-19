#Session maintain in flask
from flask import Flask, session
import os
*session maintain
if (session.get("demo") == None):
        session["demo"] = "value"
        print("This is a new session.")
    else:
        print("This is an existing session.")
if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run(host=<ip>,debug=True, port=<port_no>, threaded=True)
