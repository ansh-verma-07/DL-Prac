# Exp5: Remove noise from MNIST images

Aim: Build an autoencoder using MNIST that takes a noisy digit image and reconstructs a cleaner image.

```bash
python exp5_autoencoder/main.py
```

Run from the project root after installing requirements. Real MNIST is downloaded once into data/MNIST. No GPU is required.

## How it works

1. Load 6000 real MNIST training images.
2. Add Gaussian noise and scattered black/white pixels to make training inputs. Keep the original clean images as targets.
3. Train a small convolutional autoencoder for 10 epochs. Its encoder learns spatial image features; its decoder reconstructs the image.
4. Evaluate on 1000 separate test images with known clean targets. Print noisy and denoised MSE: lower is better.
5. Read your already-noisy image, prepare its size/colours, and reconstruct it directly. No extra noise is added to your input file.

This uses the denoising option in the practical aim. It does not claim to compress an image into 32 numbers. Digit labels are unused during training.

## Your input

Example using the included real MNIST sample with added noise:

```text
Enter NOISY digit image path (PNG/JPG): exp5_autoencoder/sample_noisy_digit.png
Digit colour (1 = dark on light, 2 = light on dark): 2
```

For a black/dark digit on white/light, choose 1. For a white/light digit on dark, choose 2. The code converts to grayscale and fits the image into 28 x 28 pixels. Output colours match your input choice. Use one digit with some surrounding space; no plot titles or axes.

For the user's noisy 3 example, enter its path ending in digits_recoverd-noise.png and choose 1. The file ending in digits_recoverd-rec.png is already a reconstructed image, not a known clean ground-truth target.

## Outputs

- Training losses for 10 epochs.
- Noisy and reconstructed MSE on held-out MNIST test images.
- denoised_digit.png: the reconstructed 28 x 28 image.
- denoising.png: your prepared noisy input and output side by side.

Open the PNGs after running. Your own image's true reconstruction error cannot be calculated without its clean original. The model can blur details or change shapes, especially when the noise differs from training; it cannot guarantee perfect recovery. Test MSE measures average performance on the specified test corruption, not on every external image.

## Data source

[MNIST through torchvision](https://docs.pytorch.org/vision/2.0/generated/torchvision.datasets.MNIST.html). sample_digit.png is the original first test image, a 7. sample_noisy_digit.png is that same real image with Gaussian noise added once (standard deviation 0.3, seed 42). Training now also includes black/white pixel noise so the model sees a wider range of corruption. These are corruptions of real images, not a generated replacement dataset.

Viva: Training uses noisy inputs paired with clean targets. Prediction requires only the noisy input. Convolution preserves spatial information. Reconstruction MSE is relevant; classification accuracy is not.


## Your included test file

Enter exp5_autoencoder/test_noisy_3.png and choose colour 1. This is an unchanged copy of your supplied noisy 3 and is included in the GitHub upload folder. reference_recovered_3.png is your previously recovered example, for viewing only, not a known clean ground truth. Neither file is used for training.

Latest held-out test: noisy MSE 0.0674, denoised MSE 0.0079 on the mixed-noise test protocol. Your noisy 3 was also visually checked: scattered noise is reduced, with the digit still recognizable as 3. This is a qualitative result, not an accuracy score for the external file.


More examples: see [samples/README.md](samples/README.md) for 20 noisy digit inputs and links to their clean MNIST references.
