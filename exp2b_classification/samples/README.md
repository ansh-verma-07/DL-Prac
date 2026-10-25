# Extra classification inputs

20 files: digits 0-9, each in light and dark versions.

Run `python exp2b_classification/main.py` from the project root. After training/testing, enter one path below and the matching colour.

| Example path | Colour choice |
|---|---|
| exp2b_classification/samples/digit_0_light.png | 2 |
| exp2b_classification/samples/digit_3_dark.png | 1 |
| exp2b_classification/samples/digit_8_light.png | 2 |

Replace the digit in the filename with any number from 0 to 9. Light means a light digit on dark background; dark means a dark digit on light background.

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
