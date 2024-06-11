from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from SportsScrapper import BCCI_Scrapper, Indian_Athletes_Scrapper, ICC_Scrapper
from NewsVerification import Validator
import numpy as np
import json

app = Flask(__name__)
CORS(app)


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
    result = json.dumps(result, cls=CustomEncoder)
    return jsonify(result)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

if __name__ == '__main__':
    app.run(debug=True)
