# Deep Learning Practical Exam

Simple Python programs for all five aims. Use Python 3.11 or 3.12. No GPU is required.

## What each experiment does

| Experiment and aim | Training and testing | Your input after testing | Output |
|---|---|---|---|
| Exp1: NumPy multilayer perceptron for binary classification | Learn the four XOR rows using manual backpropagation | Two bits, e.g. 0 then 1 | XOR class and truth-table accuracy |
| Exp2(a): PyTorch regression | Train on 436 real housing rows, test on 109 | Area, bedrooms, bathrooms | House price; test MAE and R2 |
| Exp2(b): PyTorch digit classification | Train on 6000 MNIST images, test on 1000 separate images | Path to one handwritten-digit image | Predicted digit; test accuracy; predictions.png |
| Exp5: MNIST denoising autoencoder | Train with noisy MNIST inputs and clean targets; test on 1000 separate images | Path to an ALREADY NOISY digit image | Denoised image; comparison PNG; test reconstruction MSE |
| Exp8: Web application using the same model | Load the weights and scaling saved by Exp2(a) | House details in the browser | Predicted price on the page |

**Exp5 does not add noise to your uploaded/input image.** It adds mixed Gaussian and black/white pixel noise only when preparing training/testing pairs from MNIST. At prediction time your noisy image goes straight into the trained autoencoder, after grayscale conversion and resizing. No clean reference image is required from you.

## Setup after cloning

Open a terminal inside the project folder containing this README.

```bash
python -m venv .venv
```

Activate it using the command for your terminal:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install -r requirements.txt
```

If Windows cannot find Python, install Python 3.12 with Add Python to PATH enabled and reopen the terminal. You can also try `py` instead of `python`. If PowerShell blocks activation, use Command Prompt or run `.venv\Scripts\python.exe` instead of `python` without activating.

## Commands, sample inputs and outputs

Run commands from the project root. Training and testing happen first. Then type your answer at each prompt and press Enter. The output numbers below are from local runs; library versions can change them slightly. Full logs are in each model folder's sample_output.txt.

### Exp1: XOR classification

```bash
python exp1_mlp/main.py
```

Enter `0`, then `1`.

```text
Epoch: 10000 Loss: 0.0011
[0 0] -> 0
[0 1] -> 1
[1 0] -> 1
[1 1] -> 0
Accuracy: 100.0 %
Enter first bit (0 or 1): 0
Enter second bit (0 or 1): 1
Predicted XOR: 1
```

The program also prints probabilities. Accuracy checks the same four truth-table rows used for training; it is not a held-out test. There are only four possible XOR inputs. This matches the XOR option in the aim.

### Exp2(a): Regression

```bash
python exp2a_regression/main.py
```

Enter area `5000`, bedrooms `3`, bathrooms `2`.

```text
Training rows: 436 | Testing rows: 109
Epoch: 500 Loss: 1.5746
Test average error: 919689
Test R2 score: 0.5565
Model saved as house_model.pt
Saved model loaded for prediction.
Dataset area range: 1650 to 16200. Outside this range is extrapolation.
Enter area: 5000
Enter bedrooms: 3
Enter bathrooms: 2
Predicted price: 6016436
```

MAE is the average absolute prediction error in original price units; lower is better. R2 is 1 for perfect predictions and 0 for the test-mean baseline; it can be negative. Regression does not use classification accuracy. Input scaling is fitted only on training rows. After saving, Exp2(a) explicitly reloads the saved weights and scaling before predicting your input. The small model uses three features and can have substantial price errors. The program does not assume a currency.

### Exp2(b): Digit classification

```bash
python exp2b_classification/main.py
```

After testing, enter the path to your digit image and select its colour: 1 for a dark digit, 2 for a light digit. To try the included real MNIST example:

```text
Test accuracy: 95.9 %
Correct predictions: 959 out of 1000
Enter digit image path (PNG/JPG): exp2b_classification/sample_digit.png
Digit colour (1 = dark on light, 2 = light on dark): 2
Predicted digit: 7
Saved prepared_input.png and predictions.png
```

Open exp2b_classification/predictions.png. An external image has no known label, so only its predicted digit is shown. The accuracy above is calculated on labeled MNIST test images, not on your input file.

### Exp5: Directly denoise a noisy image

```bash
python exp5_autoencoder/main.py
```

Give it an already-noisy image. A noisy example is included:

```text
Epoch: 10 Loss: 0.0079
Testing on 1000 unseen images:
Noisy image MSE: 0.0674
Denoised image MSE: 0.0079
Test noise: Gaussian noise plus scattered black/white pixels.
Enter NOISY digit image path (PNG/JPG): exp5_autoencoder/sample_noisy_digit.png
Digit colour (1 = dark on light, 2 = light on dark): 2
Saved denoised_digit.png and denoising.png
No extra noise was added to your image.
Your image has no clean reference, so its reconstruction error is unknown.
```

- denoised_digit.png: the reconstructed 28 x 28 image.
- denoising.png: your prepared noisy input and reconstructed output side by side.

Open those files in exp5_autoencoder. The convolutional encoder learns spatial image features, and the decoder reconstructs a 28 x 28 image. This experiment demonstrates denoising, not a 32-number representation. The measured test MSE uses known clean MNIST targets; lower is better. We cannot calculate a true denoising error for your own image without its clean original. No classification accuracy is applicable.

The model uses a small convolutional autoencoder trained for 10 epochs on Gaussian noise mixed with scattered black/white pixels. Reconstructions may be blurry and it may not improve every image, especially with different noise or handwriting. It is not a general photo-restoration tool.

### Exp8: Web prediction

Run Exp2(a) first to create its saved model. Finish its input prompts, then start the web app:

```bash
python exp2a_regression/main.py
python exp8_webapp/app.py
```

Open http://127.0.0.1:5000. Enter area `5000`, bedrooms `3`, bathrooms `2`, then click Predict.

```text
Predicted price: 6016436
```

The app loads the exact Exp2(a) weights, uses its saved scaling and converts the price back from millions. It does not train a second model. Keep both folders together. Stop the server with Ctrl+C. If the saved model is missing, the program asks you to run Exp2(a) first.

## Preparing your own images

For Exp2(b), use one handwritten digit with padding on a plain background. Select **1** for a dark digit on a light drawing area, or **2** for a light digit on a dark background. A yellow digit on a dark purple background is also a light digit: choose 2. The code converts colors to grayscale, removes a thin 4% outer frame on large images, crops empty space, fits the digit into 20 x 20 pixels, and centers its ink in a black 28 x 28 image. Keep the digit clear of the image edges.

The classifier now uses two small convolution layers to learn image shapes. Check prepared_input.png: it should show only a light digit on black. Do not use screenshots containing plot titles, axes, or predicted labels. The MNIST accuracy does not guarantee every personal image will be correct.

For Exp5, use an already-noisy digit. Choose 1 for dark-on-light or 2 for light-on-dark. It converts your file to grayscale and fits it into 28 x 28 pixels, then directly reconstructs it. It uses your colour choice rather than guessing from noisy pixels, and restores the original polarity in the saved output. If you only have a clean image, use the provided noisy sample to demonstrate the denoising aim.

Both accept PNG/JPG paths, for example:

```text
C:\Users\Lenovo\Desktop\my_digit.png
```

Quotes around the path are accepted. Relative paths are resolved from your terminal's current folder. Missing/unreadable files give a retry prompt. Image numbers are not required. Each run replaces the previous output PNGs.

## All dataset sources

| Used by | Source | Included/downloaded |
|---|---|---|
| Exp1 | Standard mathematical XOR truth table | Four rows in main.py; no dataset download |
| Exp2(a), Exp8 | [Kaggle Housing Prices Dataset by Yasser H](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset) | All 545 records in exp2a_regression/Housing.csv |
| Exp2(b), Exp5 | [MNIST through torchvision](https://docs.pytorch.org/vision/2.0/generated/torchvision.datasets.MNIST.html) | Automatically downloaded into data/MNIST |

Housing.csv was downloaded unchanged from this [public mirror of the Kaggle dataset](https://github.com/MartinJessenne/Housing-Price-Prediction-Kaggle-Dataset/blob/main/Housing.csv). It has 13 columns. We use area, bedrooms and bathrooms to predict price. See the source for dataset terms and provenance. The housing records are not generated examples.

MNIST is the handwritten-digit dataset by Yann LeCun, Corinna Cortes and Christopher Burges, with 60000 training and 10000 test images. Torchvision downloads the real data from its [MNIST mirror](https://ossci-datasets.s3.amazonaws.com/mnist/). The four files contain training/test images and labels. Both experiments train on the first 6000 training images and evaluate the first 1000 test images for quick CPU runs. The whole dataset is downloaded, even though the programs use subsets.

sample_digit.png in the image experiment folders is the unmodified first MNIST test image, depicting a 7. Exp5's sample_noisy_digit.png is a deliberately corrupted copy of that real image: Gaussian noise with standard deviation 0.3, clipped to 0-1, with a fixed seed of 42. Its clean source is retained for provenance. Noise is simulated for denoising practice; the source digit and training data are real MNIST records.

## Internet and GitHub upload

Internet is required for package installation and the first MNIST download. Both experiments reuse the same data/MNIST cache. Later runs on that computer work offline. A fresh clone needs a download again, or copy your existing data/MNIST folder to the exam computer.

Upload the contents of GITHUB_UPLOAD, or extract DL_Practicals_GitHub.zip and upload its contents. Keep README.md at the repository root.

```text
exp1_mlp/
exp2a_regression/       (includes Housing.csv)
exp2b_classification/  (includes sample_digit.png)
exp5_autoencoder/      (includes sample_noisy_digit.png)
exp8_webapp/           (includes templates/index.html)
data/                 (only README.md)
README.md
requirements.txt
.gitignore
```

GitHub browser drag-and-drop does not apply .gitignore. Do not upload the entire original DL folder, .test_packages, .venv, downloaded MNIST, the saved .pt model, or the ZIP itself. The prepared upload folder excludes these. Enable hidden files if .gitignore is not visible. Exp2(a) recreates the saved model after cloning.

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Replace the URL with your repository and follow Setup above.

## Remember the training steps

Load data -> make model -> calculate prediction -> calculate loss -> zero_grad() -> backward() -> step() -> evaluate -> accept your input.

Exp1 calculates gradients manually in NumPy. The PyTorch experiments use automatic differentiation. Exp8 only performs prediction with the saved model.


Exp2(b) also saves prepared_input.png: check that it contains a light digit on a black background. Supply only the digit image, not a screenshot containing titles, axes or a predicted label. Preprocessing helps match MNIST, but cannot guarantee every prediction is correct.


The housing dataset area range is 1650 to 16200. Outside this range, predictions are extrapolation and can be unreliable. Being inside the range does not guarantee an accurate price for an individual house.


## Aim verification and fixes

- Exp1: checked all four XOR predictions. Manual NumPy backpropagation and the hidden layer satisfy the binary-classification aim. The reported 100% is truth-table accuracy, not an unseen-data benchmark.
- Exp2(a): verified the 436/109 training/test split, training-only scaling, MAE and R2. The code develops a regression model; it does not promise exact property prices.
- Exp2(b): verified held-out MNIST accuracy of 95.9%, the sample 7, and the user's original 4.jpg and 3.png as predictions 4 and 3. Those personal images were used only for input testing, not training. The model and input preparation were corrected; other handwriting can still be misclassified.
- Exp5: verified that inference uses the supplied image after grayscale, colour normalization and resizing, with no extra random noise. Checked that the saved reconstruction matches the model output. Known clean references are used only for training/test metrics. This fixes the previous clean-input demonstration mismatch.
- Exp8: verified that saved weights and scaling match Exp2(a), and both paths return the same prediction for 5000, 3, 2. Checked the form and invalid-input responses.

The latest main.py/app.py files and documentation in GITHUB_UPLOAD are copies of the verified project files. Downloaded MNIST, local dependencies and model checkpoints are excluded from the upload package.


## Your included test images: no Downloads path needed

Run commands from the project root. These are unchanged copies of the images you supplied, used for prediction tests only, not added to the training dataset.

| Experiment | Enter this image path | Digit colour |
|---|---|---|
| Exp2(b) | exp2b_classification/test_4.jpg | 2 (light digit on dark) |
| Exp2(b) | exp2b_classification/test_3.png | 2 (light digit on dark) |
| Exp5 | exp5_autoencoder/test_noisy_3.png | 1 (dark digit on light) |

exp5_autoencoder/reference_recovered_3.png is your already-recovered example, included for viewing/comparison only. It is not the noisy input and is not a verified clean ground truth. Do not use it to claim a denoising error score.

The updated Exp5 uses 10 training epochs and mixed Gaussian plus black/white pixel noise. On 1000 held-out MNIST images, noisy MSE was 0.0674 and reconstructed MSE was 0.0079. This test corruption differs from the earlier Gaussian-only experiment, so the old and new MSE values are not a controlled model comparison. The denoising.png currently included shows the result on your supplied noisy 3.

These test files are included in GITHUB_UPLOAD and the ZIP, so the same relative paths work after cloning. Use the files as prediction inputs; they are not part of MNIST training.


## More sample inputs

- Exp2(a): [10 real house-detail examples](exp2a_regression/sample_inputs.md), with values to enter, recorded prices and model predictions. They also work in Exp8.
- Exp2(b): [20 digit image inputs](exp2b_classification/samples/README.md), covering 0-9 in both colours.
- Exp5: [20 already-noisy digit inputs](exp5_autoencoder/samples/README.md), covering 0-9 in both colours.

For image samples, light files use colour choice 2; dark files use choice 1. Example paths: exp2b_classification/samples/digit_8_light.png and exp5_autoencoder/samples/noisy_8_dark.png. All samples and their source information are included in GITHUB_UPLOAD and the ZIP.
