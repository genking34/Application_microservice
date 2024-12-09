from flask import Flask, request, jsonify
import torch
import numpy as np
import os

PORT = int(os.getenv("PORT", 8000))  # Use the PORT environment variable, default to 8000

# Import your model definition
class RegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Load the trained PyTorch model
input_dim = 8  # Update this to match the input dimension from training
model = RegressionModel(input_dim)
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "<h1>Bathini Akash's ML API</h1><p>Use the /predict endpoint to send JSON data for prediction.</p>"

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to get predictions from the model."""
    try:
        # Get the input JSON request
        data = request.get_json()
        # Extract features from the request and ensure it has 8 features
        features = np.array(data['features']).reshape(1, -1)
        if features.shape[1] != 8:  # Validate input dimension
            raise ValueError(f"Expected 8 features, but got {features.shape[1]} features.")
        
        # Convert to a PyTorch tensor
        features_tensor = torch.FloatTensor(features)
        # Get prediction from the model
        prediction = model(features_tensor).detach().numpy().tolist()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)  # Use host 0.0.0.0 and the PORT variable