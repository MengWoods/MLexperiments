import tensorflow as tf
import numpy as np

# --- 1. Configuration and Hyperparameters ---

# Dimensions
INPUT_DIM = 64     # D: Feature size of each input step (e.g., word embedding size)
RNN_UNITS = 32     # H: Size of the hidden state (RNN memory). This is the width of each layer.
SEQ_LENGTH = 10    # T: Length of the input sequence (number of time steps)
BATCH_SIZE = 4     # B: Number of samples processed per step
NUM_LAYERS = 2     # L: We are now explicitly defining two stacked RNN layers

# Training Parameters
LEARNING_RATE = 0.01
EPOCHS = 5         # Number of full passes over the simulated data

print(f"Configuration: D={INPUT_DIM}, H={RNN_UNITS}, T={SEQ_LENGTH}, B={BATCH_SIZE}, L={NUM_LAYERS}\n")

# --- 2. Initialize Trainable Weights for TWO LAYERS ---

# --- LAYER 1 Parameters (L1) ---
# Layer 1 takes the raw input X_t (shape D)
# W_xh_L1 (Input to Hidden) Shape: (D, H) -> [64, 32]
W_xh_L1 = tf.Variable(
    tf.random.uniform([INPUT_DIM, RNN_UNITS], minval=-0.1, maxval=0.1),
    name="L1_W_xh"
)

# W_hh_L1 (Hidden to Hidden, Recurrent) Shape: (H, H) -> [32, 32]
W_hh_L1 = tf.Variable(
    tf.random.uniform([RNN_UNITS, RNN_UNITS], minval=-0.1, maxval=0.1),
    name="L1_W_hh"
)

# b_h_L1 (Hidden layer bias) Shape: (H,) -> [32]
b_h_L1 = tf.Variable(
    tf.zeros([RNN_UNITS]),
    name="L1_b_h"
)


# --- LAYER 2 Parameters (L2) ---
# Layer 2 takes the hidden state of L1 (shape H) as its input
# W_xh_L2 (Input from L1 to Hidden L2) Shape: (H, H) -> [32, 32]
# Note: The input to L2 is h_1,t, which has size H, so the weight matrix must be (H, H)
W_xh_L2 = tf.Variable(
    tf.random.uniform([RNN_UNITS, RNN_UNITS], minval=-0.1, maxval=0.1),
    name="L2_W_xh"
)

# W_hh_L2 (Hidden to Hidden, Recurrent) Shape: (H, H) -> [32, 32]
W_hh_L2 = tf.Variable(
    tf.random.uniform([RNN_UNITS, RNN_UNITS], minval=-0.1, maxval=0.1),
    name="L2_W_hh"
)

# b_h_L2 (Hidden layer bias) Shape: (H,) -> [32]
b_h_L2 = tf.Variable(
    tf.zeros([RNN_UNITS]),
    name="L2_b_h"
)


# --- Output Layer Parameters (Uses the final layer's hidden state, L2) ---
OUTPUT_DIM = 1 # Simple output dimension
W_out = tf.Variable(
    tf.random.uniform([RNN_UNITS, OUTPUT_DIM], minval=-0.1, maxval=0.1),
    name="W_out"
)
b_out = tf.Variable(
    tf.zeros([OUTPUT_DIM]),
    name="b_out"
)


# Aggregate all trainable variables
trainable_variables = [
    W_xh_L1, W_hh_L1, b_h_L1,
    W_xh_L2, W_hh_L2, b_h_L2,
    W_out, b_out
]

print("--- Initialized Parameters ---")
print(f"Total Trainable Parameters: {sum(v.shape.num_elements() for v in trainable_variables)}\n")


# --- 3. Core RNN and Prediction Functions ---

@tf.function
def rnn_layer_step(x_t_input, h_t_minus_1, W_xh, W_hh, b_h):
    """
    Performs the recurrent calculation for one time step 't' for a SINGLE layer.

    Args:
        x_t_input: The input for this layer at time t (either x_t or h_{l-1, t}).
        h_t_minus_1: The hidden state of this layer from the previous time step (t-1).
        W_xh, W_hh, b_h: The trainable weights specific to this layer.
    """
    # Recurrent Calculation: h_t = tanh( (x_t_input @ W_xh) + (h_{t-1} @ W_hh) + b_h )

    # 1. Contribution from the input at time t (vertical connection)
    input_contribution = tf.matmul(x_t_input, W_xh)

    # 2. Contribution from the previous hidden state (horizontal recurrence)
    recurrent_contribution = tf.matmul(h_t_minus_1, W_hh)

    # 3. Combine and apply activation (tanh)
    pre_activation = input_contribution + recurrent_contribution + b_h
    h_t = tf.tanh(pre_activation)

    return h_t

@tf.function
def forward_pass(x_seq):
    """
    Runs the 2-Layer Stacked RNN over the entire sequence.
    """
    # 1. Initialize hidden states for BOTH layers (h_0)
    # Both layers start with an all-zero hidden state of shape (B, H)
    h_t_L1 = tf.zeros((BATCH_SIZE, RNN_UNITS))
    h_t_L2 = tf.zeros((BATCH_SIZE, RNN_UNITS))

    # We will store the final hidden state of the top layer (L2)
    final_h = h_t_L2

    # 2. Iterate through the sequence (TIME DIMENSION) - This is the HORIZONTAL flow
    for t in tf.range(SEQ_LENGTH):
        # Extract the input for the current time step t
        x_t = x_seq[:, t, :] # Shape: (B, D)

        # --- VERTICAL PROCESSING AT TIME t (Layer-by-Layer) ---

        # 2a. LAYER 1: Processes the raw sequence input (x_t)
        # L1 uses its own previous state (h_t_L1) for recurrence
        h_t_L1 = rnn_layer_step(
            x_t, h_t_L1, W_xh_L1, W_hh_L1, b_h_L1
        )
        # The output h_t_L1 now becomes the input for Layer 2 at this same time step t.

        # 2b. LAYER 2: Processes the output of Layer 1 (h_t_L1)
        # L2 uses its own previous state (h_t_L2) for recurrence
        h_t_L2 = rnn_layer_step(
            h_t_L1, h_t_L2, W_xh_L2, W_hh_L2, b_h_L2
        )
        # The computation for token x_t is now COMPLETE in the vertical stack.
        # The loop proceeds to the next time step (t+1) to process x_{t+1}.

        # Keep track of the final hidden state of the top layer (L2)
        if t == SEQ_LENGTH - 1:
            final_h = h_t_L2


    # 3. Final Output Layer: Use the final hidden state of the TOP LAYER (h_T_L2)
    # Y_pred = (h_T_L2 @ W_out) + b_out
    prediction = tf.matmul(final_h, W_out) + b_out

    return prediction, final_h

# --- 4. Loss Function and Optimizer ---

# Mean Squared Error
loss_fn = tf.keras.losses.MeanSquaredError()

# The optimizer adjusts the weights based on the gradients
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

# --- 5. The Training Step (Forward + Backward Pass) ---

@tf.function
def train_step(x_batch, y_true):
    """
    Performs one full training step: Forward Pass, Loss, Backward Pass (BPTT), and Optimization.
    """
    # tf.GradientTape records all operations on the trainable variables.
    with tf.GradientTape() as tape:
        # 1. FORWARD PASS
        y_pred, final_h = forward_pass(x_batch)

        # 2. LOSS CALCULATION
        loss = loss_fn(y_true, y_pred)

    # 3. BACKWARD PASS (Backpropagation Through Time - BPTT)
    # Computes d(loss)/d(variable) for all 8 trainable variables.
    gradients = tape.gradient(loss, trainable_variables)

    # 4. OPTIMIZER STEP
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, gradients, y_pred

# --- 6. Data Simulation and Training Loop ---

# Simulate input data (e.g., embedded sentences)
# Shape: (Batch, Time, Features) -> (4, 10, 64)
X_train = tf.constant(
    np.random.rand(BATCH_SIZE, SEQ_LENGTH, INPUT_DIM).astype(np.float32)
)

# Simulate target labels (e.g., scores from 0 to 1)
# Shape: (Batch, Output) -> (4, 1)
Y_train = tf.constant(
    np.random.rand(BATCH_SIZE, OUTPUT_DIM).astype(np.float32)
)

print(f"Simulated Input Shape: {X_train.shape} | Simulated Target Shape: {Y_train.shape}")
print("\n--- Starting Training ---")

# TRAINING LOOP
for epoch in tf.range(EPOCHS):
    # Perform the full training step
    current_loss, current_gradients, y_pred = train_step(X_train, Y_train)

    # Logging and Inspection
    if (epoch + 1) % 1 == 0:
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  Loss: {current_loss.numpy():.6f}")

        # Log a sample of the prediction
        print(f"  Sample True Y: {Y_train[0].numpy()}")
        print(f"  Sample Pred Y: {y_pred[0].numpy()}")

        # Log the gradient magnitude for inspection (L1 weights are the first 3 variables)
        print(f"  Gradient Norm (L1_W_xh): {tf.norm(current_gradients[0]).numpy():.4f}")
        print(f"  Gradient Norm (L2_W_xh): {tf.norm(current_gradients[3]).numpy():.4f}") # L2 is index 3

print("\n--- Training Complete ---")

# Final check of the forward pass
final_pred, final_h_state = forward_pass(X_train)
final_loss = loss_fn(Y_train, final_pred)

print(f"Final Loss after {EPOCHS} epochs: {final_loss.numpy():.6f}")
