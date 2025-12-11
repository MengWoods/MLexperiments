import numpy as np

# --- 1. Synthetic Data Generation ---
np.random.seed(42)
X_reg = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y_reg = 1 + 2 * X_reg + np.random.randn(100, 1) * 0.5  # Linear relation + noise

# --- 2. Model Parameters and Hyperparameters ---
# Initial parameters (will be printed in the loop)
w = np.random.randn(1, 1)[0][0]
b = np.random.randn(1, 1)[0][0]
learning_rate = 0.01
n_iterations = 5 # Limit to 5 iterations for detailed printout
n_samples = len(X_reg)

# --- 3. Single Neuron (Linear) Training ---
print("--- Linear Regression (Single Neuron) Detailed Training ---")

# Extract the first data point for scalar demonstration
x1 = X_reg[0, 0]
y1 = y_reg[0, 0]

print(f"Initial w = {w:.6f}, Initial b = {b:.6f}")
print(f"--- Demonstration using the first data point (x₁={x1:.4f}, y₁={y1:.4f}) ---\n")

for iteration in range(1, n_iterations + 1):

    # Store old values for printing the update step
    w_old = w
    b_old = b

    print(f"================== ITERATION {iteration} ==================")

    # ----------------------------------------------------
    # FORWARD PASS (Vectorized)
    # ----------------------------------------------------
    y_hat_all = w * X_reg + b
    error_all = y_hat_all - y_reg

    # Calculate Cost Function Value (MSE)
    cost = np.mean(error_all**2)

    # Print details for the first sample (x1, y1)
    y_hat1 = w * x1 + b
    error1 = y_hat1 - y1

    print(f"1. FORWARD PASS (y = {w:.6f}x + {b:.6f})")
    print(f"   * Predicted y₁ (ŷ₁): {y_hat1:.6f}")
    print(f"   * Real y₁:         {y1:.6f}")
    print(f"   * Error₁ (ŷ₁ - y₁): {error1:.6f}")
    print(f"   * Total Cost (J):  {cost:.6f} (Mean Squared Error across all {n_samples} samples)")


    # ----------------------------------------------------
    # BACKWARD PASS (Vectorized)
    # ----------------------------------------------------

    # dJ/dw = (2/n) * sum(error * X)
    dw = (2 / n_samples) * np.sum(error_all * X_reg)

    # dJ/db = (2/n) * sum(error)
    db = (2 / n_samples) * np.sum(error_all)

    print("\n2. GRADIENTS (Average of All Samples)")
    print(f"   * Partial dJ/dw: {dw:.6f}")
    print(f"   * Partial dJ/db: {db:.6f}")

    # ----------------------------------------------------
    # GRADIENT DESCENT UPDATE
    # ----------------------------------------------------

    w_update = learning_rate * dw
    b_update = learning_rate * db

    w = w - w_update
    b = b - b_update

    print(f"\n3. UPDATE STEP (α = {learning_rate})")
    print(f"   * w_new = w_old - α * dJ/dw")
    print(f"   * w_new = {w_old:.6f} - {learning_rate} * {dw:.6f} = {w:.6f}")
    print(f"   * b_new = b_old - α * dJ/db")
    print(f"   * b_new = {b_old:.6f} - {learning_rate} * {db:.6f} = {b:.6f}")

    # ----------------------------------------------------
    # NEW NEURAL FUNCTION
    # ----------------------------------------------------
    print(f"\n4. NEURAL FUNCTION AFTER UPDATE")
    print(f"   * Function: ŷ = {w:.6f}x + {b:.6f}")
    print("==============================================\n")

print("--- Training completed for the detailed view (5 iterations) ---")
print(f"Final Learned Parameters: w={w:.6f}, b={b:.6f}")
