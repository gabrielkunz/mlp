import numpy as np
from perceptron import Perceptron

def fit(mlp, X, target_prediction, epochs = 100):
    # Perform the fit outside Perceptron class since the epoch is no applied on
    # the entire net and the error does not come from a single neuron

    # Add bias column
    X = np.c_[X, np.ones((X.shape[0]))]

    # Loop the epochs
    for epoch in np.arange(0, epochs):
        # Loop each data input
        for (x, target) in zip(X, target_prediction):
            # Weighted input
            weighted_x = np.dot(x, mlp.weight)

            # Get the prediction
            p = mlp.activation(weighted_x)

            # Update the weight based on the error, if the error is 0, the weight will remain the same
            if p != target:
                error = p - target
                mlp.weight += -mlp.alpha * x * error

X_and = np.array([[0,0],
                  [0,1],
                  [1,0],
                  [1,1]])
y_and = np.array([[0],
                  [0],
                  [0],
                  [1]])

# Define perceptron AND and train it
print("Training perceptron for AND dataset...")
mlp = Perceptron(X_and.shape[1], alpha = 0.1)
fit(mlp, X_and, y_and, epochs = 1000)

# Display test result after training
print("Testing perceptron after training...")
for (x, target) in zip(X_and, y_and):
	prediction = mlp.predict(x)
	print("Input = {}, Target = {}, Prediction = {}".format(x, target[0], prediction))