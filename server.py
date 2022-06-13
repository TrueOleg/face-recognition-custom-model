from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO
from training_module import TrainingModule
from engineio.payload import Payload
import os
import time

training_module = TrainingModule()

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
    # socketio.emit('training-process', {'flag': 'training finished', '_id': data['model']['_id'], 'status': 'Training finished'})



@app.route('/')
def hello_world():
    socketio.emit('test', 'data')
    return 'Hello, World!'


if __name__ == '__main__':
    socketio.run(app, debug=True, port=4444, host='0.0.0.0')