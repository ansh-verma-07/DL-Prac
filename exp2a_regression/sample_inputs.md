# Sample inputs for Exp2(a): house-price prediction

Run from the project root:

```bash
python exp2a_regression/main.py
```

After training/testing, enter Area, Bedrooms, Bathrooms one at a time. Choose any row below. These are existing records from Housing.csv, not generated house data. Dataset row numbers below start at 1 and exclude the header.

| Example | Dataset row | Area | Bedrooms | Bathrooms | Recorded price | Model prediction |
|---|---|---|---|---|---|---|
| 1 | 1 | 7420 | 4 | 2 | 13,300,000 | 7,329,534 |
| 2 | 61 | 6000 | 3 | 2 | 7,140,000 | 6,786,169 |
| 3 | 121 | 6540 | 3 | 1 | 5,950,000 | 5,137,248 |
| 4 | 181 | 4500 | 4 | 2 | 5,215,000 | 5,313,633 |
| 5 | 241 | 3840 | 3 | 1 | 4,585,000 | 4,111,976 |
| 6 | 301 | 4079 | 3 | 1 | 4,200,000 | 4,283,569 |
| 7 | 361 | 4040 | 2 | 1 | 3,710,000 | 3,609,280 |
| 8 | 421 | 4120 | 2 | 1 | 3,360,000 | 3,616,091 |
| 9 | 481 | 3480 | 3 | 1 | 2,940,000 | 3,853,509 |
| 10 | 545 | 3850 | 3 | 1 | 1,750,000 | 4,119,155 |

Example: choose the Area, Bedrooms and Bathrooms from one row, then type each value at its prompt and press Enter. These same triples work in the Exp8 webpage.

The recorded price is the actual CSV value; the model prediction is an estimate using only three features. Differences are expected. These selected CSV examples may overlap the training rows and are input demonstrations, not a new held-out accuracy test. Use the program's held-out MAE and R2 for evaluation. Predictions may vary slightly with versions or retraining changes.

Source: [Kaggle Housing Prices Dataset by Yasser H](https://www.kaggle.com/datasets/yasserh/housing-prices-dataset), included as Housing.csv. Prices use the dataset's original units; no currency is assumed.
