# Exp2(b): MNIST digit classification using PyTorch

Aim: Develop a classification model that predicts handwritten digits from 0 to 9.

After installing requirements, run from the project root:

```bash
python exp2b_classification/main.py
```

## Training and testing

Load real MNIST images automatically into data/MNIST. First download requires internet. Train on 6000 training images and evaluate 1000 separate test images.

The network has two small convolution layers, ReLU and pooling, followed by a linear layer with 10 scores. Convolution learns image shapes. CrossEntropyLoss compares scores to the true digit labels. Adam updates the weights. Argmax selects the highest-scoring digit.

The local test run correctly classified 959 of 1000 images: 95.9% accuracy. Results may differ slightly between versions. This is MNIST test accuracy, not a guarantee for every image you draw or download.

## Your input

After testing, enter a PNG/JPG path, then choose the digit's colour:

```text
Enter digit image path (PNG/JPG): exp2b_classification/sample_digit.png
Digit colour (1 = dark on light, 2 = light on dark): 2
```

- Choose 1 for a black/dark digit on a white/light drawing area.
- Choose 2 for a white/light digit on black/dark. Yellow on dark purple is also light-on-dark.

Use a full path for your own file, such as C:\Users\Lenovo\Downloads\4.jpg. Quotes are accepted. Invalid paths give a retry prompt. Use one digit only and keep some space around it. Avoid photographs, transparent backgrounds and screenshots with titles or axes.

Image preparation: grayscale -> remove thin outer frame on large images -> use the chosen polarity -> crop empty space -> fit into 20 x 20 pixels -> center the ink inside a 28 x 28 black image. Removing a 4% frame helps avoid treating thin image borders as part of a digit; keep your digit away from the edge.

Output:
- Training losses and held-out test accuracy.
- Predicted digit for your file.
- prepared_input.png: exactly the prepared image sent to the model.
- predictions.png: that image with the predicted digit in the title.

Open the PNGs after running. An external file has no known label, so the program does not print an invented actual digit or individual accuracy. If it predicts incorrectly, inspect prepared_input.png and check your colour choice. Even a properly prepared image can be misclassified.

## Dataset source

[MNIST through torchvision](https://docs.pytorch.org/vision/2.0/generated/torchvision.datasets.MNIST.html). sample_digit.png is the unchanged first MNIST test image (digit 7), not a generated image. See the main README for download links and offline instructions.

Viva: Classification predicts a category. Convolution detects image features. Pooling reduces image size. Testing uses images that were not used to update the weights.


## Your included test files

Enter exp2b_classification/test_4.jpg or exp2b_classification/test_3.png from the project root. Choose colour 2 for both. These are unchanged copies of your provided images, used for prediction testing only. Verified predictions: 4 and 3. They are included in the GitHub upload folder.


More examples: see [samples/README.md](samples/README.md) for digits 0-9, in both light and dark versions (20 files).
