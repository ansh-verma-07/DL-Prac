# Exp2(a): House-price regression using PyTorch

**Aim:** Predict a continuous house price using a neural network.

From the repository root, after installing `requirements.txt`:

```bash
python exp2a_regression/main.py
```

**Dataset:** Included `Housing.csv`, 545 original rows with 13 columns. Source: [Kaggle Housing Prices Dataset by Yasser H](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset). Downloaded unchanged from this [public CSV mirror](https://github.com/MartinJessenne/Housing-Price-Prediction-Kaggle-Dataset/blob/main/Housing.csv). No Kaggle login or download is needed to run this experiment. See the source page for dataset terms and provenance; no claim is made that these records are verified property valuations.

For short code, use only area, bedrooms and bathrooms to predict price. The other CSV columns are retained but unused. Prices are reported in the dataset's original units; the program does not assume a currency.

Steps: shuffle with a fixed seed -> split 80% training / 20% test -> standardize inputs using training statistics -> divide prices by one million -> train a 3 -> 16 -> 1 network with ReLU, Adam and MSE.

Output: training loss every 100 epochs, test mean absolute error (MAE), test R2 score, one actual/predicted pair, and `house_model.pt`. This file contains trained weights and input scaling values for Exp8. Predictions need not exactly match actual prices.

**Viva:** Regression predicts a number. MSE averages squared errors. MAE averages absolute errors in original price units. Scaling makes training easier; test data is not used to fit scaling or weights.




## User input

After training/testing, the program saves the model, explicitly reloads its weights and scaling, and prints the dataset area range. Then enter area, bedrooms and bathrooms at the prompts. Example: 5000, then 3, then 2. Sample predicted price: 6016436.


More examples: see [sample_inputs.md](sample_inputs.md) for 10 real CSV records and their model predictions.
