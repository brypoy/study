import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os

current_working_dir = os.getcwd()
save_dir_relative = os.path.join(current_working_dir, "graphs")
os.makedirs(save_dir_relative, exist_ok=True)
#############################################################################################################################################################################################
#
# linear regression model
#
#############################################################################################################################################################################################
# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Create a linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_dim=1)
])

# Compile the model
model.compile(optimizer='sgd',  # Stochastic Gradient Descent
              loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, verbose=0)

# Make predictions
X_new = np.array([[0], [2]])
# Extend the prediction line to cover a larger range
X_new = np.arange(-1, 3, 0.1).reshape(-1, 1)
# make predictions
y_pred = model.predict(X_new)
#
#
############## SAVE PLOT AND PRINT ########################################################
save_path = os.path.join(save_dir_relative, "linear_regression_plot.png")
# Plot the original data and the regression line
plt.scatter(X, y, label='Original data')
plt.plot(X_new, y_pred, 'b-', label='Regression line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
# Save the plot
plt.savefig(save_path)
# Display the plot
#plt.show()

#############################################################################################################################################################################################
#
# simple neural network model
#
#############################################################################################################################################################################################
# Generate random data for illustration
np.random.seed(0)
X = np.random.randn(100, 2)  # 100 points with 2 features each
y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Class 1 if sum of features is positive, else Class 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential([
    Dense(units=4, activation='relu', input_dim=2),  # Hidden layer with 4 neurons and ReLU activation
    Dense(units=1, activation='sigmoid')  # Output layer with 1 neuron and Sigmoid activation for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.1, verbose=0)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
#
#
############## SAVE PLOT AND PRINT ########################################################
# Specify the directory relative to the current working directory
save_path = os.path.join(save_dir_relative, "neural_network_plot.png")
# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# Save the plot
plt.savefig(save_path)
# Display the plot
#plt.show()
