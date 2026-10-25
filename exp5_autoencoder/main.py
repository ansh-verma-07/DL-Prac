from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

torch.manual_seed(42)
torch.set_num_threads(2)
folder = Path(__file__).resolve().parent

# Load real digit images. Use 6000 images to train quickly.
train = MNIST(folder.parent / 'data', train=True, download=True, transform=ToTensor())
test = MNIST(folder.parent / 'data', train=False, download=True, transform=ToTensor())
loader = DataLoader(Subset(train, range(6000)), batch_size=64, shuffle=True)

# Encoder learns image features; decoder rebuilds the image.
model = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=2, padding=1), nn.ReLU(),
    nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(),
    nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1), nn.Sigmoid())
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train with both fuzzy noise and scattered black/white pixels.
def add_noise(clean):
    noisy = (clean + 0.2 * torch.randn_like(clean)).clamp(0, 1)
    spots = torch.rand_like(clean)
    noisy[spots < 0.05] = 0
    noisy[spots > 0.95] = 1
    return noisy

print('Training on 6000 images...')
for epoch in range(10):
    total = 0
    for images, labels in loader:
        clean = images
        noisy = add_noise(clean)
        output = model(noisy)
        loss = loss_function(output, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    print('Epoch:', epoch + 1, 'Loss:', round(total / len(loader), 4))

# Test with known clean targets, so we can measure denoising error.
model.eval()
test_loader = DataLoader(Subset(test, range(1000)), batch_size=100)
before, after = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        clean = images
        noisy = add_noise(clean)
        before += loss_function(noisy, clean).item()
        after += loss_function(model(noisy), clean).item()
print('\nTesting on 1000 unseen images:')
print('Noisy image MSE:', round(before / len(test_loader), 4))
print('Denoised image MSE:', round(after / len(test_loader), 4))
print('Test noise: Gaussian noise plus scattered black/white pixels.')

# Give the trained model an ALREADY NOISY image. Do not add noise here.
while True:
    path = input('Enter NOISY digit image path (PNG/JPG): ').strip().strip('"')
    try:
        with Image.open(path) as file:
            image = file.convert('L')  # Change to grayscale.
        break
    except (OSError, ValueError):
        print('Cannot open the image. Enter a valid PNG or JPG path.')

# Tell the model which colour is the digit; noisy corners are unreliable.
colour = input('Digit colour (1 = dark on light, 2 = light on dark): ').strip()
if colour == '1':
    image = ImageOps.invert(image)
elif colour != '2':
    raise SystemExit('Choose 1 for a dark digit or 2 for a light digit.')
image = ImageOps.pad(image, (28, 28), color=0)
noisy = ToTensor()(image).unsqueeze(0)
with torch.no_grad():
    output = model(noisy)

# Display and save in the same colours as the user's input.
if colour == '1':
    noisy, output = 1 - noisy, 1 - output
result = Image.fromarray((output.view(28, 28).numpy() * 255).round().astype('uint8'))
result.save(folder / 'denoised_digit.png')
plt.figure(figsize=(6, 3))
pictures = [noisy, output]
titles = ['Your noisy input', 'Denoised output']
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(pictures[i].view(28, 28), cmap='gray', vmin=0, vmax=1)
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.savefig(folder / 'denoising.png')
plt.close()
print('Saved denoised_digit.png and denoising.png')
print('No extra noise was added to your image.')
print('Your image has no clean reference, so its reconstruction error is unknown.')
