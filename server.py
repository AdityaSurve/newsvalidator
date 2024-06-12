from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from SportsScrapper import BCCI_Scrapper, Indian_Athletes_Scrapper, ICC_Scrapper
from NewsVerification import Validator
from flask.json import JSONEncoder
import numpy as np
class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super(CustomJSONEncoder, self).default(obj)

app = Flask(__name__)
CORS(app)
app.json_encoder = CustomJSONEncoder


@app.route('/bcci', methods=['GET'])
def get_bcci():
    player_name = request.args.get('player_name')
    platform = request.args.get('platform')
    type = request.args.get('type')

    scrapper = BCCI_Scrapper()
    articles = scrapper.get_player_data(player_name, platform, type)

    return jsonify(articles)


@app.route('/indianathletics', methods=['GET'])
def get_indianathletics():
    player_name = request.args.get('player_name')

    scrapper = Indian_Athletes_Scrapper()
    articles = scrapper.get_player_data(player_name)

    return jsonify(articles)


@app.route('/icc', methods=['GET'])
def get_icc():
    player_name = request.args.get('player_name')

    scrapper = ICC_Scrapper()
    articles = scrapper.get_player_data(player_name)

    return jsonify(articles)


@app.route('/validator', methods=['GET'])
def get_cricket():
    player_name = request.args.get('player_name')
    platform = request.args.get('platform')
    type = request.args.get('type')
    news_type = request.args.get('news_type')
    validator = Validator()
    result = validator.search(player_name, type, platform, news_type)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
