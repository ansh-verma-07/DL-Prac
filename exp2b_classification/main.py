from pathlib import Path
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

torch.manual_seed(42)
torch.set_num_threads(2)
folder = Path(__file__).resolve().parent

# Load real digit images and select small groups for a quick run.
train = MNIST(folder.parent / 'data', train=True, download=True, transform=ToTensor())
test = MNIST(folder.parent / 'data', train=False, download=True, transform=ToTensor())
train_loader = DataLoader(Subset(train, range(6000)), batch_size=64, shuffle=True)
test_loader = DataLoader(Subset(test, range(1000)), batch_size=64)

# Two small convolution layers learn digit shapes, then predict 10 scores.
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 7 * 7, 10))
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print('Training on 6000 images...')
for epoch in range(5):
    total = 0
    for images, labels in train_loader:
        output = model(images)
        loss = loss_function(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print('Epoch:', epoch + 1, 'Loss:', round(total / len(train_loader), 4))

# Count correct answers on unseen images.
model.eval()
correct = 0
print('\nTesting on 1000 unseen images...')
with torch.no_grad():
    for images, labels in test_loader:
        prediction = model(images).argmax(1)
        correct += (prediction == labels).sum().item()
print('Test accuracy:', round(correct / 1000 * 100, 2), '%')
print('Correct predictions:', correct, 'out of 1000')

# Read your own handwritten digit image.
while True:
    path = input('Enter digit image path (PNG/JPG): ').strip().strip('"')
    try:
        with Image.open(path) as file:
            image = file.convert('L')  # Change to grayscale.
        break
    except (OSError, ValueError):
        print('Cannot open the image. Enter a valid PNG or JPG path.')

# Remove a thin outer frame from large images. Keep the digit away from edges.
if min(image.size) > 56:
    margin = int(min(image.size) * 0.04)
    image = image.crop((margin, margin, image.width - margin, image.height - margin))

# Ask about the digit colour instead of guessing from a border.
colour = input('Digit colour (1 = dark on light, 2 = light on dark): ').strip()
if colour == '1':
    # Remove dark outer padding around a light drawing area.
    paper = image.point(lambda pixel: 255 if pixel > 127 else 0).getbbox()
    if paper is not None:
        image = image.crop(paper)
    image = ImageOps.invert(image)
elif colour != '2':
    raise SystemExit('Choose 1 for a dark digit or 2 for a light digit.')

# Crop empty space, then center the digit like an MNIST image.
box = image.point(lambda pixel: 255 if pixel > 50 else 0).getbbox()
if box is None:
    raise SystemExit('No digit found. Use a clear handwritten digit image.')
digit = ImageOps.pad(image.crop(box), (20, 20), color=0)
image = Image.new('L', (28, 28), color=0)
image.paste(digit, (4, 4))
# Center the ink, as MNIST does, rather than only centering its box.
pixels = np.array(image, dtype=float)
rows, cols = np.indices(pixels.shape)
dx = round(13.5 - (cols * pixels).sum() / pixels.sum())
dy = round(13.5 - (rows * pixels).sum() / pixels.sum())
centered = Image.new('L', (28, 28), color=0)
centered.paste(image, (dx, dy))
image = centered
image.save(folder / 'prepared_input.png')
image = ToTensor()(image)
with torch.no_grad():
    answer = model(image.unsqueeze(0)).argmax(1).item()
print('Predicted digit:', answer)
plt.figure()
plt.imshow(image[0], cmap='gray')
plt.title('Predicted digit: ' + str(answer))
plt.axis('off')
plt.savefig(folder / 'predictions.png')
plt.close()
print('Saved prepared_input.png and predictions.png')
