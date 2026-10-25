# Exp8: Simple prediction website

Aim: Enter house details in a webpage and get a price from the Exp2(a) model.

Run these commands from the main project folder after installing requirements:

```bash
python exp2a_regression/main.py
python exp8_webapp/app.py
```

Open http://127.0.0.1:5000 in your browser.
Enter area 5000, bedrooms 3 and bathrooms 2. Click Predict.
Press Ctrl+C in the terminal to stop the website.

Files:
- app.py: loads the saved model and calculates the price.
- templates/index.html: contains the simple input form.

Exp2(a) must run first to create house_model.pt. Keep both folders together.
If you get a missing house_model.pt error, run Exp2(a) first.
The app uses the exact saved weights and scaling from Exp2(a).
Prices use the dataset's original units.

Viva: Flask connects the webpage to Python. GET opens the page. POST sends the form values. Training learns the weights; prediction uses the saved weights. This is a basic local exam demo; the browser checks that required inputs are positive.
