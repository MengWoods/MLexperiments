# Basic

## RNN

[two layer RNN nets](./src/two_layer_RNN.py)

- Recurrence: The hidden state from the previous time step is fed back into the network along with the current input to produce the new hidden state.
  - Recurrence Equation: $h_t = tanh( (x_t @ W_xh) + (h_{t-1} @ W_hh) + b_h )$
  - Hyperbolic tangent function is used as the major activation for the hidden state if RNN layer, because:
    - The output range is centered at zero $[-1, 1]$. It helps keep the data normalized, leading to faster and more stable convergence during training. In contrast the sigmoid function's output is $[0,1]$, which is not zero-centered and can bias the gradient updates.
    - The derivative of tanh is generally larger than the derivative of sigmoid, a larger gradient helps the weight updates flow more effectively during backpropagation through time, which makes the network learn faster and somewhat mitigates the vanishing gradient problem compared to sigmoid.
- Unrolling the network: visualize an RNN as a deep feed-forward network where each time step is a separate layer.
- Backpropagation through time (BRTT): The training algorithm used for RNNs.
  -
- The vanishing gradient problem: The primary limitation of standard RNNs, as the sequence gets longer, the initial information's gradient shrinks to zero, making it impossible for the network to learn long-term dependencies.

A basic RNN model typically involves three main layer types:
- Embedding layer: converts input tokens into dense numerical vectors.
- SimpleRNN Layer: The core layer that processes the sequence step by step and maintains a hidden state (memory).
- Dense Layer: The output layer that maps the RNN's output to a prediction.

## LSTM

LSTMs were explicitly designed to solve the vanishing gradient problem in RNNs, making them the default choice for most sequence tasks for many years.

- The Cell State: This is the conveyor belt or the long-term memory. It runs straight through the unit, only having linear interactions, which allos gradients to flow easily.
- The three gates: These are the internal mechanisms that regulate information flow into and out of the cell state. They are typically sigmoid layers that output a value between 0 and 1, controlling how much information is let through.
  - Forget Gate: Decides what information to throw away from the old cell state.
  - Input Gate: Decides what new information from the current input should be stored in the cell state.
  - Output Gate: Decides what part of the cell state should be passed to the hidden state, which is the final output of the unit.

## GRUs

Gated Recurrent Units, two-gate version of LSTMs, faster to train.
