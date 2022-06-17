from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO
from predict_module import PredictModule
from training_module import TrainingModule
from engineio.payload import Payload
import os
import time

training_module = TrainingModule()
predict_module = PredictModule()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)

Payload.max_decode_packets = 500
socketio = SocketIO(app, cors_allowed_origins='*')


@socketio.on('upload_files')
def test(data):
    print('data', data['model'])
    socketio.emit('training-process', {'flag': 'in training', '_id': data['model']['_id'], 'status': 'Training in process'})
    training_module.training_function(data['files'], data['model'])
    socketio.emit('training-process', {'flag': 'training finished', '_id': data['model']['_id'], 'status': 'Training finished'})


@app.route('/predict', methods=["POST"])
def predict():

    file = request.files['file']
    file_path = os.path.join('./tmp', file.filename)
    file.save(file_path)
    predict_module.predict(file_path, request.form['model_id'])
    return 'Hello, World!'


if __name__ == '__main__':
    socketio.run(app, debug=False, port=4444, host='0.0.0.0')