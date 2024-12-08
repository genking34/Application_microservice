# Importing required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import torch  # PyTorch library for building and training the neural network
import torch.nn as nn  # For defining neural network modules
from torch.utils.data import Dataset, DataLoader, random_split  # For dataset handling
from sklearn.preprocessing import MinMaxScaler  # For normalizing data
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.metrics import mean_absolute_error, r2_score  # For evaluating model performance

# # Load the dataset directly from a URL
# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/concrete.csv"
# data = pd.read_csv(url)


import pandas as pd

# Load the dataset from the UCI Machine Learning Repository
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
data = pd.read_excel(url)

# Display the first few rows
print(data.head())




# Display the first few rows of the dataset for inspection
print(data.head())

# Define features (inputs) and target (output)
X = data.iloc[:, :-1].values  # Select all columns except the last one as features
y = data.iloc[:, -1].values   # Select the last column as the target (compressive strength)

# Normalize the feature data to a range of [0, 1] using Min-Max Scaling
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert the features and target into PyTorch tensors for model training
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)  # Features as float tensors
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # Target as float tensors (unsqueeze adds a dimension)

# Split the dataset into training and testing sets using an 80-20 split
dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)  # Combine features and target into a single dataset
train_size = int(0.8 * len(dataset))  # Calculate the size of the training set
test_size = len(dataset) - train_size  # Calculate the size of the testing set
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])  # Perform the split

# Create data loaders for batch processing during training and testing
batch_size = 32  # Number of samples per batch
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle training data for randomness
test_loader = DataLoader(test_dataset, batch_size=batch_size)  # No need to shuffle test data




# Define the neural network model for regression
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the regression model with fully connected layers and ReLU activation.
        Args:
            input_dim: Number of input features
        """
        super(RegressionModel, self).__init__()  # Initialize the parent class
        self.fc = nn.Sequential(  # Define the model architecture
            nn.Linear(input_dim, 64),  # First layer with 64 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(64, 32),  # Second layer with 32 neurons
            nn.ReLU(),  # Activation function
            nn.Linear(32, 1)  # Output layer with a single neuron for regression
        )

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor
        Returns:
            Output tensor
        """
        return self.fc(x)  # Pass input through the defined layers



# Initialize the model, loss function, and optimizer
input_dim = X_tensor.shape[1]  # Determine the number of input features
model = RegressionModel(input_dim)  # Create an instance of the model
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with a learning rate of 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)




# Train the model for a specified number of epochs
num_epochs = 60  # Number of iterations over the entire dataset
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0  # Initialize epoch loss
    for inputs, targets in train_loader:  # Loop over each batch of data
        optimizer.zero_grad()  # Clear gradients from the previous step
        outputs = model(inputs)  # Perform a forward pass
        loss = criterion(outputs, targets)  # Calculate the loss
        loss.backward()  # Backpropagate the gradients
        optimizer.step()  # Update the model parameters
        epoch_loss += loss.item()  # Accumulate the loss for the epoch

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")



# Evaluate the model on the test set
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # Disable gradient computation for evaluation
    y_true, y_pred = [], []  # Lists to store true and predicted values
    for inputs, targets in test_loader:  # Loop over the test data batches
        outputs = model(inputs)  # Perform a forward pass
        y_true.extend(targets.numpy())  # Collect true values
        y_pred.extend(outputs.numpy())  # Collect predicted values

# Convert the true and predicted values to NumPy arrays for metric calculation
y_true = np.array(y_true).flatten()  # Flatten the array for compatibility
y_pred = np.array(y_pred).flatten()  # Flatten the array for compatibility

# Calculate evaluation metrics
mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
r2 = r2_score(y_true, y_pred)  # R² score to measure goodness of fit

# Print the evaluation metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Display some sample predictions for inspection
print("Sample predictions:")
for true, pred in zip(y_true[:10], y_pred[:10]):  # Loop over the first 10 samples
    print(f"True: {true:.2f}, Predicted: {pred:.2f}")

import torch

# Assuming 'model' is your trained PyTorch model
torch.save(model.state_dict(), './model.pth')

print("Model saved successfully as 'model.pth'")

