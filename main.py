from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('cleaned_car.csv')

@app.route("/")
def index():
    companies = sorted(car['company'].unique())
    car_models_dict = {}
    for comp in companies:
        car_models_dict[comp] = sorted(car[car['company'] == comp]['name'].unique())
    year = sorted(car['year'].unique())
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template("index.html",
                           companies=companies,
                           car_models=car_models_dict,
                           years=year,
                           fuel_types=fuel_type,
                           prediction=None,
                           selected_company=None,
                           selected_model=None,
                           selected_year=None,
                           selected_fuel=None,
                           selected_kilo=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        company = request.form.get('company')
        car_model = request.form.get('car_model')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        kms_driven = int(request.form.get('kilo_driven'))
        print(company, car_model, year, kms_driven, fuel_type)

        # Correct column order
        df = pd.DataFrame(
            [[car_model, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )

        # Prediction
        prediction_arr = model.predict(df)
        pred_value = float(prediction_arr[0])

        return jsonify({'prediction': pred_value})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
