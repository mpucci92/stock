from flask import Flask, render_template
import os

import src.algo.RandomForestClassifier_McGill_v1
from src.algo.RandomForestClassifier_McGill_v1 import main_function
import requests
template_dir = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__,template_folder=template_dir)

@app.route('/', methods=['GET'])
def hello():
    return f'Hello you should use an other route:!\nEX: get_stock_val/<ticker>\n'


@app.route('/get_stock_val/<ticker>', methods=['GET'])
def get_stock_value(ticker):
    prediction = main_function(str(ticker))[0]
    
    return render_template("index.html",title=prediction)
    


if __name__ == '__main__':
    # Used when running locally only. When deploying to Cloud Run,
    # a webserver process such as Gunicorn will serve the app.
    app.run(host='127.0.0.1', port=8080, debug=True)
    