from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Load the trained model
model_path = "model.pkl"
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not found!"}), 500
    
    try:
        data = request.json['features']
        input_features = np.array(data).reshape(1, -1)
        prediction = model.predict(input_features)[0]
        result = "Rock" if prediction == 1 else "Mine"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
