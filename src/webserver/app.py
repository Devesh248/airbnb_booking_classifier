import json
import sys
from flask import Flask, request
from src.common.constants import *
from src.prediction.prediction import Prediction

app = Flask(__name__)


predictor = Prediction()


@app.route('/')
def hello_world():
    return json.dumps({"message": "This is ML based classifier"})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()['data']
    n_data = request.get_json()['n_data']
    result = predictor.single_prediction_handler(data, n_data)
    response = {
       "prediction": result
    }
    return json.dumps(response)


@app.route('/predict_batch', methods=['POST'])
def batch_predict():
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(prediction_data_dir, filename)
    file.save(file_path)
    result = predictor.batch_prediction_handler(file_path)
    response = {
       "prediction": result
    }
    return json.dumps(response)


# main driver function
if __name__ == '__main__':
    app.run(debug=True)
