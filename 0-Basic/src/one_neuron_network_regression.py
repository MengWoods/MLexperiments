import numpy as np

# --- 1. Synthetic Data Generation ---
# Y = 2X + 1 + noise
np.random.seed(42)
X_reg = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y_reg = 1 + 2 * X_reg + np.random.randn(100, 1) * 0.5  # Linear relation + noise

# --- 2. Model Parameters and Hyperparameters ---
w = np.random.randn(1, 1)[0][0]  # Initialize weight randomly
b = np.random.randn(1, 1)[0][0]  # Initialize bias randomly
learning_rate = 0.01
n_iterations = 1000
n_samples = len(X_reg)

# --- 3. Single Neuron (Linear) Training ---
print("--- Linear Regression (Single Neuron) Training ---")

for iteration in range(n_iterations):
    # Forward Pass: Prediction
    # y_hat = w * X + b  (Vectorized)
    y_hat = w * X_reg + b

    # Calculate Error
    error = y_hat - y_reg

    # Backward Pass: Calculate Gradients
    # dJ/dw = (2/n) * sum(error * X)
    dw = (2 / n_samples) * np.sum(error * X_reg)

    # dJ/db = (2/n) * sum(error)
    db = (2 / n_samples) * np.sum(error)

    # Gradient Descent Update
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Optional: Print loss every 100 iterations
    if iteration % 100 == 0:
        cost = np.mean(error**2)
        print(f"Iteration {iteration:4d}: Cost = {cost:.4f}, w = {w:.3f}, b = {b:.3f}")

# --- 4. Testing the Learned Model ---
print("\n--- Testing the Learned Model ---")
X_test = np.array([[0.5], [1.5], [2.5]])
y_test_pred = w * X_test + b
print(f"Final Learned Parameters: w={w:.4f}, b={b:.4f}")
print("Test Input (X_test): \n", X_test.flatten())
print("Predicted Output (y_hat): \n", y_test_pred.flatten())
