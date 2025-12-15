import numpy as np

# --- 1. Define Vectors and Hyperparameters ---
# Assume an input sequence of 4 elements, with an embedding dimension of 8
seq_len = 4
d_k = 8  # Dimension of the Key and Query vectors

# Synthesize Keys and Values (K and V come from the ENCODER output)
# K: (seq_len, d_k) = (4, 8)
# V: (seq_len, d_k) = (4, 8)
np.random.seed(42)
K = np.random.randn(seq_len, d_k)
V = np.random.randn(seq_len, d_k)

# Synthesize Query (Q comes from the DECODER or the current token)
# Q: (1, d_k) = (1, 8)
Q = np.random.randn(1, d_k)

print("--- Initial Parameters and Vectors ---")
print(f"Key Matrix (K) shape: {K.shape}")
print(f"Value Matrix (V) shape: {V.shape}")
print(f"Query Vector (Q) shape: {Q.shape}")
print("-" * 40)

# --- 2. The Scaled Dot-Product Attention Function ---

def scaled_dot_product_attention(Q, K, V):
    """
    Calculates the attention mechanism output.

    Q: Query vector(s)
    K: Key matrix
    V: Value matrix
    """
    # ----------------------------------------------------
    # Step 1: Calculate Alignment Scores (Query-Key Dot Product)
    # ----------------------------------------------------
    # Scores = Q * K^T
    # Resulting shape: (1, d_k) * (d_k, seq_len) = (1, seq_len)

    # The dot product measures how well the Query aligns with each Key.
    scores = np.dot(Q, K.T)

    print(f"Step 1: Alignment Scores (Q * K^T): {scores.shape}")
    print(f"  Scores: {scores.flatten()}")


    # ----------------------------------------------------
    # Step 2: Scaling
    # ----------------------------------------------------
    # Divide by the square root of the key dimension (d_k)
    scaling_factor = np.sqrt(d_k)
    scores_scaled = scores / scaling_factor

    # This scaling prevents the dot product from becoming too large, which
    # could push the softmax gradient to near-zero (vanishing gradient problem).
    print(f"Step 2: Scaled Scores (div by √d_k={scaling_factor:.2f}): {scores_scaled.flatten()}")


    # ----------------------------------------------------
    # Step 3: Softmax (Attention Weights)
    # ----------------------------------------------------
    # Softmax turns the scores into probabilities (weights) that sum to 1.
    # We use a standard stable softmax implementation: exp(x) / sum(exp(x))

    # Subtracting max for numerical stability (doesn't change result)
    max_score = np.max(scores_scaled)
    exp_scores = np.exp(scores_scaled - max_score)

    attention_weights = exp_scores / np.sum(exp_scores)

    print(f"Step 3: Attention Weights (Softmax): {attention_weights.shape}")
    print(f"  Weights (Sum={np.sum(attention_weights):.4f}): {attention_weights.flatten()}")


    # ----------------------------------------------------
    # Step 4: Context Vector (Weighted Sum of Values)
    # ----------------------------------------------------
    # Context = Weights * V
    # Resulting shape: (1, seq_len) * (seq_len, d_k) = (1, d_k)

    # We multiply the weights by the Value matrix (V).
    # The higher the weight for an input element, the more its Value contributes
    # to the final context vector.
    context_vector = np.dot(attention_weights, V)

    print(f"Step 4: Context Vector (Weights * V): {context_vector.shape}")

    return context_vector, attention_weights


# --- 3. Execute the Attention Calculation ---
context, weights = scaled_dot_product_attention(Q, K, V)

print("\n--- Final Result ---")
print("The Context Vector is the result of 'attending' to the input.")
print(f"Context Vector (Shape {context.shape}):")
print(context)

# Interpretation
highest_weight_index = np.argmax(weights)
print(f"\nHighest Attention Weight is at index {highest_weight_index} with a value of {weights.flatten()[highest_weight_index]:.4f}.")
print("This means the model decided the Key/Value pair at this index was the most relevant to the current Query.")
