# Gradient Descend

Gradient Descent is an iterative optimization algorithm used to minimize a cost function by adjusting model parameters in the direction of the steepest descent of the function's gradient. It find the optimal values of weights and biases by gradually reducing the error between predicted and actual outputs.

## Mathematics

Cost function: $J(\omega, b) = \frac{1}{n} \sum_{i=1}^{n} (y)$



## Technique for Efficient Training


- Weights regularzations: Initialization of weights in an appropriate range.
- Activation function: using activation function such as ReLU can help to mitigate the vanishing gradient problem.
- Gradient clipping: Restrict the gradients to a predefined range to prevent them from becoming excessively large or small.
- Batch normalization: Normalizing the input of each layer to prevent activation function from saturating and hence reducing vanishing the exploding gradient problems.
