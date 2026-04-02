"""
House Price Prediction — Flask Backend
"""
import json, pickle, warnings
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_from_directory

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ── Load model & metadata ─────────────────────────────────────────────────────
with open('models/model.pkl', 'rb') as f:
    MODEL = pickle.load(f)

with open('models/metadata.json') as f:
    META = json.load(f)

PREMIUM  = {'Bandra', 'Juhu', 'Worli', 'Lower Parel', 'Dadar'}
MID      = {'Andheri', 'Powai', 'Goregaon'}


def location_tier(loc):
    if loc in PREMIUM:  return 'premium'
    if loc in MID:      return 'mid'
    return 'affordable'


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html', meta=META)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        area      = float(data['area'])
        bedrooms  = int(data['bedrooms'])
        bathrooms = int(data['bathrooms'])
        parking   = int(data.get('parking', 1))
        floor_no  = float(data.get('floor', 5))
        age       = float(data.get('age', 5))
        location  = str(data['location'])
        furnished = str(data['furnished'])

        room_ratio  = bathrooms / max(bedrooms, 1)
        total_rooms = bedrooms + bathrooms
        loc_tier    = location_tier(location)

        row = pd.DataFrame([{
            'Area': area, 'Bedrooms': bedrooms, 'Bathrooms': bathrooms,
            'Parking': parking, 'Floor': floor_no, 'Age_of_Property': age,
            'Room_ratio': room_ratio, 'Total_rooms': total_rooms,
            'Location': location, 'Location_tier': loc_tier,
            'Furnished_Status': furnished
        }])

        log_price = MODEL.predict(row)[0]
        price     = np.expm1(log_price)

        return jsonify({
            'success': True,
            'price':   round(float(price), 2),
            'price_lakhs': round(float(price) / 1e5, 2),
            'price_crores': round(float(price) / 1e7, 3),
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/meta')
def meta():
    return jsonify(META)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
