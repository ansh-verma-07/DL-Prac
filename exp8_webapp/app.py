from pathlib import Path
import math
import torch
from torch import nn
from flask import Flask, request, render_template

# Load the model saved in Exp2(a). Run Exp2(a) first.
folder = Path(__file__).resolve().parent
if not (folder.parent / 'exp2a_regression' / 'house_model.pt').exists():
    raise SystemExit('Run python exp2a_regression/main.py first to save the model.')
saved = torch.load(folder.parent / 'exp2a_regression' / 'house_model.pt',
                   map_location='cpu', weights_only=True)
model = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 1))
model.load_state_dict(saved['weights'])
model.eval()
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    result = ''
    if request.method == 'POST':
        # Read the three numbers entered in the form.
        try:
            area = float(request.form['area'])
            bedrooms = int(request.form['bedrooms'])
            bathrooms = int(request.form['bathrooms'])
            if not (math.isfinite(area) and 0 < area <= 1000000
                    and 1 <= bedrooms <= 6 and 1 <= bathrooms <= 4):
                raise ValueError
        except (ValueError, KeyError, OverflowError):
            return render_template('index.html', result='Enter valid house details.'), 400
        x = torch.tensor([[area, bedrooms, bathrooms]])
        x = (x - saved['mean']) / saved['std']
        with torch.no_grad():
            price = model(x).item() * 1000000
        result = 'Predicted price: ' + str(round(price))
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(port=5000)
