from flask import Flask, request
from flask_cors import CORS, cross_origin

from summarizer import Summarizer
from utility import Utility
import json

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route("/api/summarization", methods=["POST"])
@cross_origin()
def summarization():
    try:
        json_data = request.get_json(force=True)

        text = json_data.get('text')
        method = json_data.get('method', 'T5')
        pretrained = json_data.get('pretrained', 't5-large')
    
        # initialize Summarizer
        s = Summarizer(method=method, pretrained=pretrained)
        
        pred = s.summarize(text)
        if isinstance(pred, dict):
            summary = pred['summary']
            data = dict()
            data['text'] = text
            data['summary'] = summary
            data['summary_len'] = Utility.get_doc_length(summary)
            data['text_len'] = Utility.get_doc_length(text)
            return json.dumps(data)
        else:
            return json.dumps(pred)

    except Exception as e:
        return {"Error": str(e)}


if __name__ == "__main__":
    app.run(debug=True, port="5000")