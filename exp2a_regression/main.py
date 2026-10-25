from pathlib import Path
import numpy as np
import math
import torch
from torch import nn

# Read the real housing dataset.
folder = Path(__file__).resolve().parent
data = np.loadtxt(folder / 'Housing.csv', delimiter=',', skiprows=1,
                  usecols=(0, 1, 2, 3), dtype=np.float32)
np.random.seed(42)
np.random.shuffle(data)
torch.manual_seed(42)
torch.set_num_threads(2)

# Inputs: area, bedrooms, bathrooms. Output: price in millions.
x = torch.tensor(data[:, 1:].copy())
y = torch.tensor(data[:, :1].copy()) / 1000000
n = int(0.8 * len(x))
mean = x[:n].mean(0)
std = x[:n].std(0)
x = (x - mean) / std

# Make a small neural network.
model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train using the first 80% of rows.
print('Training rows:', n, '| Testing rows:', len(x) - n)
for epoch in range(500):
    prediction = model(x[:n])
    loss = loss_function(prediction, y[:n])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch:', epoch + 1, 'Loss:', round(loss.item(), 4))

# Test using the remaining rows.
model.eval()
with torch.no_grad():
    prediction = model(x[n:])
    error = (prediction - y[n:]).abs().mean() * 1000000
    r2 = 1 - ((prediction - y[n:]) ** 2).sum() / ((y[n:] - y[n:].mean()) ** 2).sum()
print('\nTesting results:')
print('Test average error:', round(error.item()))
print('Test R2 score:', round(r2.item(), 4))
print('First actual price:', round(y[n].item() * 1000000))
print('First predicted price:', round(prediction[0].item() * 1000000))

# Save the model and scaling values for Exp8.
torch.save({'weights': model.state_dict(), 'mean': mean, 'std': std},
           folder / 'house_model.pt')
print('Model saved as house_model.pt')

# Load the saved weights and scaling before making a new prediction.
saved = torch.load(folder / 'house_model.pt', map_location='cpu', weights_only=True)
model.load_state_dict(saved['weights'])
mean = saved['mean']
std = saved['std']
model.eval()
print('Saved model loaded for prediction.')

# Predict the price for your own house details.
print('Dataset area range: 1650 to 16200. Outside this range is extrapolation.')
try:
    area = float(input('Enter area: '))
    bedrooms = int(input('Enter bedrooms: '))
    bathrooms = int(input('Enter bathrooms: '))
except ValueError:
    raise SystemExit('Please enter numeric house details.')
if math.isfinite(area) and 0 < area <= 1000000 and 1 <= bedrooms <= 6 and 1 <= bathrooms <= 4:
    user_input = torch.tensor([[area, bedrooms, bathrooms]])
    user_input = (user_input - mean) / std
    with torch.no_grad():
        price = model(user_input).item() * 1000000
    print('Predicted price:', round(price))
else:
    print('Use area above 0 up to 1000000, 1-6 bedrooms and 1-4 bathrooms.')
