# Extra noisy inputs for Exp5

20 noisy files: digits 0-9, each in light and dark versions. These files already contain noise: give them directly to the program.

Run `python exp5_autoencoder/main.py` from the project root. After training/testing, enter a path and the matching colour:

| Example path | Colour choice |
|---|---|
| exp5_autoencoder/samples/noisy_0_light.png | 2 |
| exp5_autoencoder/samples/noisy_3_dark.png | 1 |
| exp5_autoencoder/samples/noisy_8_light.png | 2 |

Replace the digit in the filename with any number from 0 to 9. Each run overwrites denoising.png and denoised_digit.png; copy or rename the output if you want to keep it.

The clean reference for noisy_5_light.png, for example, is exp2b_classification/samples/digit_5_light.png. Matching dark references are there too. These are known clean references for these supplied MNIST samples only, unlike arbitrary user photos.

Corruption: Gaussian noise with standard deviation 0.2 plus 5% black and 5% white pixel replacements, clipped to 0-1; NumPy random seed 2026. Light and dark files for each digit show the same corruption with opposite colours. Reconstruction can still blur details.

## Source

These are real MNIST test images, one per digit, selected from test indices 1000 onwards (outside the first 1000 used in the scripts' reported test metrics). They are not added to training. Labels come from the original MNIST test labels. A known label does not guarantee a correct model prediction.

Source: [Torchvision MNIST](https://docs.pytorch.org/vision/2.0/generated/torchvision.datasets.MNIST.html). The light clean files preserve the original 28 x 28 grayscale pixels. Dark versions invert the colours. No new digit shapes were generated.

| Digit | Original MNIST test index |
|---|---|
| 0 | 1001 |
| 1 | 1004 |
| 2 | 1002 |
| 3 | 1020 |
| 4 | 1010 |
| 5 | 1003 |
| 6 | 1014 |
| 7 | 1006 |
| 8 | 1007 |
| 9 | 1000 |
