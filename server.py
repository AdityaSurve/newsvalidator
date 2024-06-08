from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from SportsScrapper import BCCI_Scrapper
from SportsScrapper import Indian_Athletes_Scrapper

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


if __name__ == '__main__':
    app.run(debug=True)
