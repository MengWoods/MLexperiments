import numpy as np

# --- 1. Synthetic Data Generation ---
# Create 2D data (2 features) that is linearly separable
np.random.seed(42)
n_samples = 100
# Class 0: centered around (1, 1)
X0 = np.random.randn(n_samples // 2, 2) * 0.5 + [1, 1]
y0 = np.zeros((n_samples // 2, 1))
# Class 1: centered around (3, 3)
X1 = np.random.randn(n_samples // 2, 2) * 0.5 + [3, 3]
y1 = np.ones((n_samples // 2, 1))

X_cls = np.vstack((X0, X1))
y_cls = np.vstack((y0, y1))

# --- 2. Activation Function ---
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- 3. Model Parameters and Hyperparameters ---
# Weights must match number of features (2)
w = np.random.randn(2, 1)
b = np.random.randn(1, 1)[0][0]
learning_rate = 0.05
n_iterations = 2000

# --- 4. Single Neuron (Logistic) Training ---
print("\n--- Binary Classification (Single Neuron) Training ---")

for iteration in range(n_iterations):
    # Forward Pass:
    # 1. Linear combination (z = X * w + b)
    z = np.dot(X_cls, w) + b
    # 2. Activation (y_hat = sigmoid(z))
    y_hat = sigmoid(z)

    # Calculate Error for Backpropagation
    # This error (y_hat - y) is the term needed for BCE gradient
    error = y_hat - y_cls

    # Backward Pass: Calculate Gradients
    # dJ/dw = (1/n) * X^T * error (Vectorized)
    dw = (1 / n_samples) * np.dot(X_cls.T, error)

    # dJ/db = (1/n) * sum(error)
    db = (1 / n_samples) * np.sum(error)

    # Gradient Descent Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Optional: Print loss and accuracy
    if iteration % 200 == 0:
        cost = -np.mean(y_cls * np.log(y_hat) + (1 - y_cls) * np.log(1 - y_hat))
        # Accuracy: threshold y_hat at 0.5
        predictions = (y_hat > 0.5).astype(int)
        accuracy = np.mean(predictions == y_cls) * 100
        print(f"Iteration {iteration:4d}: Cost = {cost:.4f}, Accuracy = {accuracy:.1f}%")

# --- 5. Testing the Learned Model ---
print("\n--- Testing the Learned Model ---")
X_test = np.array([[1.2, 1.0], [3.5, 3.2], [2.0, 2.0]]) # Test near boundary too
z_test = np.dot(X_test, w) + b
y_test_prob = sigmoid(z_test)
y_test_pred = (y_test_prob > 0.5).astype(int)

print(f"Final Learned Parameters: w1={w[0, 0]:.4f}, w2={w[1, 0]:.4f}, b={b:.4f}")
print("Test Input (X): \n", X_test)
print("Predicted Probabilities: \n", y_test_prob.flatten())
print("Predicted Classes: \n", y_test_pred.flatten())
