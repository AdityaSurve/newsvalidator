from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from SportsScrapper import BCCI_Scrapper, Indian_Athletes_Scrapper, ICC_Scrapper
from AgricultureScrapper import ICAR_Scrapper, FAO_Scrapper

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


@app.route('/icar', methods=['GET'])
def get_icar():
    query = request.args.get('query')

    scrapper = ICAR_Scrapper()
    articles = scrapper.get_query_data(query)

    return jsonify(articles)


@app.route('/fao', methods=['GET'])
def get_fao():
    query = request.args.get('query')

    scrapper = FAO_Scrapper()
    articles = scrapper.get_query_data(query)

    return jsonify(articles)

if __name__ == '__main__':
    app.run(debug=True)
