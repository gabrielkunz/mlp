import numpy as np
from mlp import MLP

# Inputs and targets
#AND
X_and = np.array([[0,0], [0,1], [1,0], [1,1]])
y_and = np.array([[0], [0], [0], [1]])

#OR
X_or = np.array([[0,0], [0,1], [1,0], [1,1]])
y_or = np.array([[0], [1], [1], [1]])

#XOR
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([[0], [1], [1], [0]])

# Select input and target
X = X_or
y = y_or

# Define perceptron AND and train it
print("Training perceptron for dataset...")
mlp = MLP(X, 1, 0.1)
mlp.fit(X, y, epochs = 1000)

# Display test result after training
print("Testing perceptron after training...")
for (x, target) in zip(X, y):
	prediction = mlp.predict(x)
	print("Input = {}, Target = {}, Prediction = {}".format(x, target[0], prediction))