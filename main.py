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
dataset = 0
while (dataset > 3 or dataset < 1):
	dataset = int(input("Select a dataset (1 - AND table, 2 - OR table, 3 - XOR table): "))

if (dataset == 1):
	X = X_and
	y = y_and
elif (dataset == 2):
	X = X_or
	y = y_or
elif (dataset == 3):
	X = X_xor
	y = y_xor

mlp = MLP(X, i = X.shape[1], k = y.shape[1])
print("Training single perceptron for dataset...")
mlp.fit_single(X, y, epochs = 1000)

print("Testing single perceptron after training...")
for (x, target) in zip(X, y):
	prediction_single = mlp.predict_single(x)
	print("Input = {}, Target = {}, Prediction = {}".format(x, target[0], prediction_single))

print("Training MLP for dataset...")
mlp.fit(X, y, epochs = 1000)
print("Testing MLP after training...")
for (x, target) in zip(X, y):
	prediction = mlp.predict(x)
	print("Input = {}, Target = {}, Prediction = {}".format(x, target[0], prediction))
