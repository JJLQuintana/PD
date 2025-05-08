from flask import Flask, jsonify
from flask_cors import CORS
from predictor import predict_dos
from datetime import datetime
app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    label = predict_dos()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return jsonify({
  "timestamp": "2025-05-09 00:49:29",
  "result": "Benign" or "DoS",
  "log": "127.0.0.1 - - [Time] \"POST /predict HTTP/1.1\" 200 -"
}
)
if __name__ == '__main__':
    app.run(debug=True, port=5000)
