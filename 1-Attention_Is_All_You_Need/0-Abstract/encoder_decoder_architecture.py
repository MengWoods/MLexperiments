import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# --- Model Parameters ---
# Dimensionality of the encoder's hidden state (the 'summary'), it defines the size of the latent space.
# Encoder reads the entire input sequence and boil it down into a single, fixed-length vector (the context vector).
# The vector is the latent representation or context vector that captures the essence of the input sequence.
# If it is 256, the final output of the encoder will be a single vector with 256 numerical values.
latent_dim = 256

# Vocabulary sizes for input and output languages
# Those values are critical as they define the size of the one-hot encoded vectors used to represent each token.
# It determines the dimensionality of the model's input and output layers and the size of its embedding matrix.
num_encoder_tokens = 71  # Vocabulary size of the input language
num_decoder_tokens = 93  # Vocabulary size of the output language

# Maximum number of tokens the encoder part of a model is designed to accept and process in a single input sequence.
max_encoder_seq_length = 16 # Max length of input sentence
max_decoder_seq_length = 20 # Max length of output sentence

# --- 1. THE ENCODER ---

# Input layer for the input sequence (e.g., an English sentence)
# encoder_inputs is a 3D tensor with shape (batch_size, max_seq_length, embedding_dim)
# It contains the sequence of dense embedding vectors for all input tokens.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
print("Encoder Input Shape:", encoder_inputs.shape)

# The core ENCODER layer: an LSTM
# 'return_state=True' is crucial: it returns the final hidden state (h) and cell state (c)
encoder_lstm = LSTM(latent_dim, return_state=True)

# encoder_outputs: a 3D tensor representing the hidden state at every single time step.
# We won't use it in the basic encoder-decoder model without attention.
# state_h: the final hidden state (context vector), 2D tensor of shape (batch_size, latent_dim)
# This vector is considered the context vector or summary of the entire input sequence. It is used to initialize the decoder's initial hidden state
# state_c: the final cell state (context vector), 2D tensor of shape (batch_size, latent_dim)
# This vector is used to initialize the decoder's initial cell state
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)

# We discard `encoder_outputs` (the step-by-step outputs) and keep the final states.
# The states (h and c) are the 'context vector' that summarizes the input sequence.
# The pairing of state_h and state_c are the two components that jointly capture the meaning of the input sentence and are passed to the decoder to begin the generation process.
encoder_states = [state_h, state_c]
print("Encoder States (Context Vector) Shape:", state_h.shape)

# --- 2. THE DECODER ---

# Input layer for the target sequence (e.g., the French sentence)
# The target sequence is fed one step ahead, typically prefixed by a 'start-of-sequence' token.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
print("Decoder Input Shape:", decoder_inputs.shape)

# The core DECODER layer: another LSTM
# 'return_sequences=True' is needed to output the full sequence for training.
# 'initial_state' is crucial: it receives the context vector from the encoder.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Final Dense (Output) Layer
# This layer maps the LSTM output to the vocabulary size (93 tokens),
# using a softmax activation to get probabilities for the next token.
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
print("Decoder Output Shape:", decoder_outputs.shape)

# --- 3. THE TRAINING MODEL ---
# This model takes the input sequence and the target sequence (shifted) and outputs the predicted sequence.
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

print("\n--- Model Summary (Structure Only) ---")
model.summary()

# --- Conceptual Testing Data ---
# In a real scenario, this data would be one-hot encoded or embedded.
# We use random data just to demonstrate the flow and shapes.

# A batch of 64 input sequences (sentences)
encoder_data = np.random.rand(64, max_encoder_seq_length, num_encoder_tokens)
# A batch of 64 target sequences (translations, including the 'start' token)
decoder_data = np.random.rand(64, max_decoder_seq_length, num_decoder_tokens)

print("\n--- Testing Data Shapes ---")
print(f"Input Data Shape: {encoder_data.shape}")
print(f"Target Data Shape: {decoder_data.shape}")

# Use the model to predict (this output is meaningless since the model isn't trained)
# This simulates passing a batch of data through the network.
dummy_predictions = model.predict([encoder_data, decoder_data])

print("\n--- Prediction Output ---")
# The output shape confirms that for 64 inputs, we get a predicted sequence
# of length 20, where each step has a probability vector over the 93 tokens.
print(f"Output Prediction Shape (Batch, Seq Length, Vocab Size): {dummy_predictions.shape}")
