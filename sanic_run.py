from sanic import Sanic
from sanic.response import json
from sanic.response import text
from sanic.exceptions import ServerError
from sanic.exceptions import NotFound
import gensim
from nltk.corpus import stopwords

app = Sanic(__name__)
model_path = "<path_to_the_model_in_server like /index/model/word2vec_model>"

@app.route("/machine_learning/model/")
async def relatedSkillsFor(request):

    try:
        model = gensim.models.word2vec.Word2Vec.load(model_path)
        myskills = request.args['s']
        myskills = [s.lower() for s in myskills]
        maxcount = request.args['maxcount'][0]
        res = model.most_similar(positive=myskills, topn=int(maxcount))
    except KeyError:
        return json({"error":"word %s not in our dictonary/vocabulary" % request.args['s']})
    return json({"similarSkills":[{"skill":item[0],"score":item[1]} for item in res]})

    
@app.exception(NotFound)
def ignore_404s(request, exception):
    return text("Word not found {}".format(request.url))

app.run(host="0.0.0.0", port=8099, debug=True, workers=1)
