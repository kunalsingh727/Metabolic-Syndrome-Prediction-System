import pickle
import numpy as np
import pandas as pd
import re
from flask import Flask, request, render_template

# --- Load artifacts ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('imputer.pkl', 'rb') as f:
    imputer = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# --- Feature columns ---
feature_columns = [
    'Age', 'Sex', 'Marital', 'Income', 'Race', 'WaistCirc', 'BMI',
    'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides',
    'WaistCirc_BMI', 'Triglycerides_HDL'
]
categorical_columns = ['Sex', 'Marital', 'Race']
numerical_columns = [col for col in feature_columns if col not in categorical_columns]

# --- Modes for categoricals ---
categorical_modes = {
    'Sex': encoders['Sex'].transform(['Female'])[0],
    'Marital': encoders['Marital'].transform(['Married'])[0],
    'Race': encoders['Race'].transform(['White'])[0]
}

# --- Clinical thresholds for qualitative mapping ---
CLINICAL_MAPPING = {
    'high': {
        'BloodGlucose': 126,
        'Triglycerides': 200,
        'BMI': 30,
        'WaistCirc': 100
    },
    'low': {
        'HDL': 40
    },
    'normal': {
        'BloodGlucose': 90,
        'Triglycerides': 150,
        'HDL': 60,
        'BMI': 25,
        'WaistCirc': 88
    }
}

# --- Natural language parser ---
def parse_natural_input(text):
    data = {
        'Age': None,
        'Sex': None,
        'Marital': None,
        'Income': None,
        'Race': None,
        'WaistCirc': 0,
        'BMI': 0,
        'Albuminuria': 0,
        'UrAlbCr': 0,
        'UricAcid': 0,
        'BloodGlucose': 0,
        'HDL': 0,
        'Triglycerides': 0
    }
    # Age
    age_match = re.search(r'(\d+)[ -]?year-old', text)
    if age_match:
        data['Age'] = int(age_match.group(1))
    # Sex
    if re.search(r'\bfemale\b', text, re.I):
        data['Sex'] = 'Female'
    elif re.search(r'\bmale\b', text, re.I):
        data['Sex'] = 'Male'
    # Marital
    marital = re.search(r'(single|married|divorced|widowed)', text, re.I)
    if marital:
        data['Marital'] = marital.group(1).capitalize()
    # Income
    income = re.search(r'income of \$?([\d,]+)', text)
    if income:
        data['Income'] = float(income.group(1).replace(',', ''))
    # Race
    for race in ['asian', 'white', 'black', 'hispanic']:
        if race in text.lower():
            data['Race'] = race.capitalize()
            break
    # Waist
    waist = re.search(r'waist.*?(\d+\.?\d*)\s*cm', text, re.I)
    if waist:
        data['WaistCirc'] = float(waist.group(1))
    # BMI
    bmi = re.search(r'BMI.*?(\d+\.?\d*)', text, re.I)
    if bmi:
        data['BMI'] = float(bmi.group(1))
    # Albuminuria
    if 'albuminuria' in text.lower():
        alb = re.search(r'albuminuria.*?(\d+)', text, re.I)
        if alb:
            data['Albuminuria'] = int(alb.group(1))
    # UrAlbCr
    uralbcr = re.search(r'UrAlbCr.*?(\d+\.?\d*)', text, re.I)
    if uralbcr:
        data['UrAlbCr'] = float(uralbcr.group(1))
    # UricAcid
    uric = re.search(r'UricAcid.*?(\d+\.?\d*)', text, re.I)
    if uric:
        data['UricAcid'] = float(uric.group(1))
    # BloodGlucose
    glucose = re.search(r'(blood glucose|blood sugar).*?(\d+\.?\d*)\s*mg/dL', text, re.I)
    if glucose:
        data['BloodGlucose'] = float(glucose.group(2))
    # HDL
    hdl = re.search(r'HDL.*?(\d+\.?\d*)\s*mg/dL', text, re.I)
    if hdl:
        data['HDL'] = float(hdl.group(1))
    # Triglycerides
    tg = re.search(r'triglycerides.*?(\d+\.?\d*)\s*mg/dL', text, re.I)
    if tg:
        data['Triglycerides'] = float(tg.group(1))

    # Handle qualitative
    if 'high blood sugar' in text.lower() and data['BloodGlucose'] == 0:
        data['BloodGlucose'] = CLINICAL_MAPPING['high']['BloodGlucose']
    if 'high triglycerides' in text.lower() and data['Triglycerides'] == 0:
        data['Triglycerides'] = CLINICAL_MAPPING['high']['Triglycerides']
    if 'low hdl' in text.lower() and data['HDL'] == 0:
        data['HDL'] = CLINICAL_MAPPING['low']['HDL']
    if 'high bmi' in text.lower() and data['BMI'] == 0:
        data['BMI'] = CLINICAL_MAPPING['high']['BMI']
    if 'high waist' in text.lower() and data['WaistCirc'] == 0:
        data['WaistCirc'] = CLINICAL_MAPPING['high']['WaistCirc']

    # Fill missing values as per your rule
    for k in ['Age', 'Sex', 'Marital', 'Income', 'Race']:
        if data[k] is None:
            data[k] = None
    for k in ['WaistCirc', 'BMI', 'Albuminuria', 'UrAlbCr', 'UricAcid', 'BloodGlucose', 'HDL', 'Triglycerides']:
        if data[k] == 0:
            data[k] = 0

    return data

# --- Prediction logic (from your logic file) ---
def predict_metabolic_syndrome(input_data):
    data = {col: input_data.get(col, np.nan) for col in feature_columns}
    for col in data:
        if data[col] == '' or data[col] is None:
            data[col] = np.nan
        else:
            if col in numerical_columns and not pd.isna(data[col]):
                try:
                    data[col] = float(data[col])
                except (ValueError, TypeError):
                    data[col] = np.nan

    input_df = pd.DataFrame([data], columns=feature_columns)
    input_df['WaistCirc_BMI'] = input_df['WaistCirc'] * input_df['BMI']
    input_df['Triglycerides_HDL'] = input_df['Triglycerides'] / (input_df['HDL'] + 1e-6)
    numerical_data = input_df[numerical_columns]
    imputed_data = imputer.transform(numerical_data)
    input_df[numerical_columns] = imputed_data
    for col in numerical_columns:
        mean = imputer.statistics_[numerical_columns.index(col)]
        lower_bound = mean - 2 * (mean / 3)
        upper_bound = mean + 2 * (mean / 3)
        input_df[col] = input_df[col].clip(lower_bound, upper_bound)
    input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
    input_df['WaistCirc_BMI'] *= 0.5
    input_df['Triglycerides_HDL'] *= 0.5
    for col in categorical_columns:
        if pd.isna(input_df[col]).iloc[0] or input_df[col].iloc[0] == '':
            input_df[col] = categorical_modes[col]
        else:
            try:
                input_df[col] = encoders[col].transform([input_df[col].iloc[0]])[0]
            except ValueError:
                input_df[col] = categorical_modes[col]
        input_df[col] = input_df[col].astype('Int64')
    threshold = 0.85
    X = input_df[feature_columns].values
    proba = model.predict_proba(X)[0, 1]
    prediction = 1 if proba >= threshold else 0
    result = 'Metabolic Syndrome Present' if prediction == 1 else 'No Metabolic Syndrome'
    return result, proba

# --- Flask app ---
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        raw_text = request.form['medical_text']
        input_data = parse_natural_input(raw_text)
        result, proba = predict_metabolic_syndrome(input_data)
        return render_template('result.html',
                               raw_input=raw_text,
                               processed_data=input_data,
                               prediction=result,
                               probability=f"{proba:.2%}")
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
