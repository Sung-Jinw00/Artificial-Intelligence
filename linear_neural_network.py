"""This neural network is linear because the relationship
	between the inputs and outputs is linear (y = x * 2)."""

import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential, layers
from keras.optimizers import SGD

# Define the model: single linear layer (no hidden layers)
model = Sequential([
	layers.Input(shape=(1,)),
	layers.Dense(1)  # no activation => linear
])

# Datasets
inputs = np.array([1, 2, 3, 4, 5], dtype=float).reshape(-1, 1)
outputs = np.array([2, 4, 6, 8, 10], dtype=float).reshape(-1, 1)

# Variables to store weights and bias
weights_list = []
biases_list = []

# Compile the model with SGD (Stochastic Gradient Descent)
# mse = mean squared error
model.compile(loss="mse", optimizer=SGD(learning_rate=0.01))

# Number of epochs for training
epochs = 2000

# Manual training loop to track the evolution of weights and bias
for epoch in range(epochs):
	model.train_on_batch(inputs, outputs)  # Train on the entire batch
	weights, biases = model.layers[0].get_weights()
	weights_list.append(weights[0][0])
	biases_list.append(biases[0])

# Display the evolution of weight and bias on a graph
plt.plot(weights_list, label="Weight")
plt.plot(biases_list, label="Bias")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Evolution of weight and bias")
plt.legend()
plt.show()

# Test the model with user input
while True:
	x = float(input("Number: "))
	pred = model.predict(np.array([[x]]), verbose=0)
	print("Prediction:", pred[0][0])
