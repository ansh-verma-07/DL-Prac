# Exp1: Multilayer Perceptron using NumPy

**Aim:** Implement binary classification for XOR using NumPy, including manual backpropagation.

From the repository root:

```bash
python -m pip install numpy
python exp1_mlp/main.py
```

Dataset is the four-row XOR truth table inside the program. No download needed.

Network: 2 inputs -> 4 sigmoid hidden neurons -> 1 sigmoid output. Output >= 0.5 means class 1. The hidden layer allows the network to learn XOR, which a single linear boundary cannot separate. Training uses binary cross-entropy gradients and gradient descent.

Expected classes: `[0,0] -> 0`, `[0,1] -> 1`, `[1,0] -> 1`, `[1,1] -> 0`. The terminal also prints probabilities and accuracy.

**Viva:** An epoch is one pass through the training data. Backpropagation calculates loss gradients; gradient descent updates the weights to reduce loss.


## User input

After the accuracy is printed, enter two bits at the prompts. Example: 0, then 1. The model prints Predicted XOR: 1.
