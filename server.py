from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from SportsScrapper import BCCI_Scrapper, Indian_Athletes_Scrapper, ICC_Scrapper

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


@app.route('/cricket', methods=['GET'])
def get_cricket():
    player_name = request.args.get('player_name')
    news_type = request.args.get('news_type')

    news_type = news_type.lower()
    articles = []

    if news_type == 'bcci':
        scrapper = BCCI_Scrapper()
        articles = scrapper.get_player_data(player_name, 'bcci', 'cricket')
    elif news_type == 'icc':
        scrapper = ICC_Scrapper()
        articles = scrapper.get_player_data(player_name)
    elif news_type == 'indianathletes':
        scrapper = Indian_Athletes_Scrapper()
        articles = scrapper.get_player_data(player_name)
    else:
        return jsonify({'error': 'Invalid news type'})

    return jsonify(articles)

if __name__ == '__main__':
    app.run(debug=True)
