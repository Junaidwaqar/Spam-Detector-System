import numpy as np
import matplotlib.pyplot as plt

# XOR dataset
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# ------------------ Activation Functions ------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ------------------ Initialize Weights ------------------
np.random.seed(1)
W1 = np.random.rand(2,2)   # Input → Hidden
b1 = np.random.rand(1,2)

W2 = np.random.rand(2,1)   # Hidden → Output
b2 = np.random.rand(1,1)

loss_history = []
acc_history  = []

# ------------------ Training Loop ------------------
for epoch in range(5000):

    # -------- Forward Propagation --------
    hidden_input  = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input  = np.dot(hidden_output, W2) + b2
    predicted    = sigmoid(final_input)

    # -------- Loss (MSE) --------
    error = y - predicted
    loss  = np.mean(error ** 2)
    loss_history.append(loss)

    # -------- Accuracy --------
    predictions = (predicted > 0.5).astype(int)
    acc = np.mean(predictions == y)
    acc_history.append(acc)

    # -------- Backpropagation --------
    d_output = error * sigmoid_derivative(predicted)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_output)

    # -------- Weight Updates --------
    W2 += hidden_output.T.dot(d_output) * 0.1
    b2 += np.sum(d_output, axis=0, keepdims=True) * 0.1

    W1 += X.T.dot(d_hidden) * 0.1
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * 0.1

# ------------------ Final Output ------------------
print("Final Predictions:")
print(predicted.round())

# ------------------ Required Plots ------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Loss Reduction")

plt.subplot(1,2,2)
plt.plot(acc_history)
plt.title("Accuracy Improvement")

plt.savefig("phase5_training_plots.png")
plt.show()

print("\nphase5_training_plots.png saved successfully.")
