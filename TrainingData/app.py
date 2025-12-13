import flask
from flask import Flask, request, jsonify
from TrainingData.data.dev_test_last_fm import *

app = Flask(__name__)

@app.route('/artist-input', methods=['POST'])
def input():
    data_to_process= request.get_json(force=True)
    artist_info_pre_json = get_artist_info(data_to_process['artist'])
    return jsonify(artist_info_pre_json)

if __name__ == '__main__':
    app.run(debug=True)