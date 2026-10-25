import numpy as np

# XOR: output is 1 only when the two inputs are different.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
np.random.seed(42)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print('Training the XOR model...')
for epoch in range(10000):
    # Forward pass: 2 inputs -> 4 hidden neurons -> 1 output.
    hidden = sigmoid(X @ W1 + b1)
    output = sigmoid(hidden @ W2 + b2)
    if (epoch + 1) % 2000 == 0:
        loss = -np.mean(y * np.log(output) + (1 - y) * np.log(1 - output))
        print('Epoch:', epoch + 1, 'Loss:', round(loss, 4))
    # Backpropagation for binary cross-entropy loss.
    d_output = (output - y) / len(X)
    d_hidden = (d_output @ W2.T) * hidden * (1 - hidden)
    W2 -= 0.5 * hidden.T @ d_output
    b2 -= 0.5 * d_output.sum(axis=0, keepdims=True)
    W1 -= 0.5 * X.T @ d_hidden
    b1 -= 0.5 * d_hidden.sum(axis=0, keepdims=True)

print('\nChecking all four XOR inputs:')
output = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
for inputs, probability in zip(X, output.ravel()):
    print(f"{inputs} -> {int(probability >= 0.5)} (probability={probability:.3f})")
print("Accuracy:", np.mean((output >= 0.5) == y) * 100, "%")

# Try your own input after training.
a = int(input('Enter first bit (0 or 1): '))
b = int(input('Enter second bit (0 or 1): '))
if a in (0, 1) and b in (0, 1):
    user_input = np.array([[a, b]])
    answer = sigmoid(sigmoid(user_input @ W1 + b1) @ W2 + b2)
    print('Predicted XOR:', int(answer.item() >= 0.5))
else:
    print('Please enter only 0 or 1.')
