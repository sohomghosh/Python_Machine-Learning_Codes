#!/usr/bin/python
from flask import request, url_for
#from flask.ext.api import FlaskAPI, status, exceptions
from flask_api import FlaskAPI, status, exceptions

app = FlaskAPI(__name__)


notes = {
    0: 'do the shopping',
    1: 'build the codez',
    2: 'paint the door',
}

def note_repr(key):
    return {
        'url': request.host_url.rstrip('/') + url_for('notes_detail', key=key),
        'text': notes[key]
    }


@app.route("/", methods=['GET', 'POST'])
def notes_list():
    """
    List or create notes.
    """
    if request.method == 'POST':
        note = str(request.data.get('text', ''))
        idx = max(notes.keys()) + 1
        notes[idx] = note
        #return note_repr(idx), status.HTTP_201_CREATED
        return str(note)+"appending more"

    # request.method == 'GET'
    return [note_repr(idx) for idx in sorted(notes.keys())]


@app.route("/<int:key>/", methods=['GET', 'PUT', 'DELETE'])
def notes_detail(key):
    """
    Retrieve, update or delete note instances.
    """
    if request.method == 'PUT':
        note = str(request.data.get('text', ''))
        notes[key] = note
        return note_repr(key)

    elif request.method == 'DELETE':
        notes.pop(key, None)
        return '', status.HTTP_204_NO_CONTENT

    # request.method == 'GET'
    if key not in notes:
        raise exceptions.NotFound()
    return note_repr(key)


if __name__ == "__main__":
    app.run(debug=True)


#Installations
#yum install python-flask
#pip3 install flask
#pip3 install flask_api

#Source 
#http://www.flaskapi.org/#roadmap

#Change permissions
#chmod +x flask_all.py

#Run
#python ./flask_all.py


#curl -X POST http://127.0.0.1:5000/ -d text="flask api is teh awesomez"
#flask api is teh awesomezappending more

#$ curl -X GET http://127.0.0.1:5000/
#[{"url": "http://127.0.0.1:5000/0/", "text": "do the shopping"}, {"url": "http://127.0.0.1:5000/1/", "text": "build the codez"}, {"url": "http://127.0.0.1:5000/2/", "text": "paint the door"}]

#$ curl -X GET http://127.0.0.1:5000/1/
#{"url": "http://127.0.0.1:5000/1/", "text": "build the codez"}

#$ curl -X PUT http://127.0.0.1:5000/1/ -d text="flask api is teh awesomez"
#{"url": "http://127.0.0.1:5000/1/", "text": "flask api is teh awesomez"}
