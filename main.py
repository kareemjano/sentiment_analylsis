import os
from flask import Flask, request, jsonify
import sys
from flask_selfdoc import Autodoc
sys.path.append('src')

from src.runners.infer import infer

app = Flask(__name__)
auto = Autodoc(app)

# inputs
training_data = os.path.join('build', 'datasets', 'labelled_text.csv')
ckpts_path = os.path.join('build', 'models')


@app.route('/list_ckpts', methods=['GET'])
@auto.doc()
def list_ckpts():
    """
    Lists available checkpoints.
    Returns:
        Dict: containing available checkpoints {ckpts: ckpt_list}
    """
    ckpts = os.listdir(ckpts_path)
    return jsonify({'ckpts': ckpts})


@app.route('/predict', methods=['POST'])
@auto.doc()
def predict():
    """
    Predicts the sentiment of the given text.
    Args:
        JSON: a json dict {text: 'input text', ckpt: 'ckpt to be used'}
    Returns:
        Dict: a dict having the predicted label and probability.
                Ex. {'confidence': 0.8330674767494202, 'label': 'Positive'}

    """
    data = request.get_json()

    text = data["text"]
    ckpt = data["ckpt"]
    ckpt = os.path.join(ckpts_path, ckpt)

    prob, pred = infer(text, ckpt=ckpt)
    return jsonify({'label': pred, 'confidence': prob})

@app.route('/doc')
@auto.doc()
def documentation():
    """
    API Documentation
    """
    return auto.html()

if __name__ == '__main__':
    port = 8080
    app.run(host='0.0.0.0', port=port, debug=True)