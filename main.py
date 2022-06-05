import os
from flask import Flask, request, jsonify
import sys
sys.path.append('src')

from src.runners.infer import infer

app = Flask(__name__)

# inputs
training_data = os.path.join('build', 'datasets', 'labelled_text.csv')
ckpts_path = os.path.join('build', 'models')


@app.route('/list_ckpts', methods=['GET'])
def list_ckpts():
    ckpts = os.listdir(ckpts_path)
    return jsonify({'ckpts': ckpts})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    text = data["text"]
    ckpt = data["ckpt"]
    ckpt = os.path.join(ckpts_path, ckpt)

    prob, pred = infer(text, ckpt=ckpt)
    return jsonify({'label': pred, 'confidence': prob})


if __name__ == '__main__':
    port = 8080
    app.run(host='0.0.0.0', port=port, debug=True)