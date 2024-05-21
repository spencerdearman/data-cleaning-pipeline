from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import logging
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hashlib import sha256

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Basic cleaning functions
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

def fix_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    if not numeric_cols.empty:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    if not non_numeric_cols.empty:
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')
        df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def label_encode(df, columns, mappings):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mappings[col] = {index: label for index, label in enumerate(le.classes_)}
    return df, mappings

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

def save_mappings_to_csv(mappings, filename):
    all_mappings = []
    for col, mapping in mappings.items():
        for index, label in mapping.items():
            all_mappings.append({'column': col, 'numeric_value': index, 'category': label})
    mappings_df = pd.DataFrame(all_mappings)
    mappings_df.to_csv(filename, index=False)

def encode_categorical_columns(df, threshold=20):
    categorical_columns = df.select_dtypes(include=['object']).columns
    mappings = {}
    for col in categorical_columns:
        if df[col].nunique() < threshold and col not in ['name', 'author', 'narrator']:
            df, mappings = label_encode(df, [col], mappings)
    return df, mappings

def find_uniform_prefixes(df):
    prefixes = {}
    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)
        if not col_values.empty:
            prefix = col_values.str.extract(r'^([^:]+:)', expand=False).mode()
            if not prefix.empty:
                prefix = prefix[0]
                if all(col_values.str.startswith(prefix)):
                    prefixes[col] = prefix
    return prefixes

def clean_uniform_prefixes(df, prefixes):
    for col, prefix in prefixes.items():
        pattern = re.compile(rf'^{re.escape(prefix)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

def find_uniform_postfixes(df):
    postfixes = {}
    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)
        if not col_values.empty:
            postfix = col_values.str.extract(r'([^:]+:)$', expand=False).mode()
            if not postfix.empty:
                postfix = postfix[0]
                if all(col_values.str.endswith(postfix)):
                    postfixes[col] = postfix
    return postfixes

def clean_uniform_postfixes(df, postfixes):
    for col, postfix in postfixes.items():
        pattern = re.compile(rf'{re.escape(postfix)}$', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

def find_uniform_substrings(df):
    substrings = {}
    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)
        if not col_values.empty:
            substring = col_values.mode()
            if not substring.empty:
                substring = substring[0]
                if all(col_values == substring):
                    substrings[col] = substring
    return substrings

def clean_uniform_substrings(df, substrings):
    for col, substring in substrings.items():
        pattern = re.compile(rf'{re.escape(substring)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

def split_caps_columns(df):
    def split_caps(text):
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, str) and ' ' in x).any():
            continue
        mask = df[col].apply(lambda x: isinstance(x, str) and bool(re.search(r'[a-z][A-Z]', x)))
        if mask.any():
            split_series = df.loc[mask, col].apply(split_caps)
            new_cols = split_series.str.split(' ', expand=True, n=1)
            df[f'{col} first'] = new_cols[0]
            df[f'{col} last'] = new_cols[1]
            df = df.drop(columns=[col])

    return df

def detect_and_remove_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std())
    df_clean = df[(z_scores < threshold).all(axis=1)]
    return df_clean

def standardize_dates(df):
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    common_date_formats = [
        '%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d',
        '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y', '%m/%d/%y', '%d/%m/%y',
        '%d-%b-%Y', '%b-%d-%Y'
    ]

    for col in date_columns:
        for fmt in common_date_formats:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt, errors='raise')
                break
            except ValueError:
                continue
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df

def remove_highly_missing_columns(df, threshold=0.5):
    missing_fraction = df.isnull().mean()
    columns_to_remove = missing_fraction[missing_fraction > threshold].index
    df = df.drop(columns=columns_to_remove)
    return df

def anonymize_columns(df, columns_to_anonymize):
    for col in columns_to_anonymize:
        df[col] = df[col].apply(lambda x: sha256(x.encode()).hexdigest() if isinstance(x, str) else x)
    return df

def analyze_and_clean(df):
    actions_taken = {}

    df = lower_case_columns(df)
    actions_taken['Lower Case Columns'] = True
    
    if df.isnull().values.any():
        df = fix_missing_values(df)
        actions_taken['Fix Missing Values'] = True

    if df.duplicated().any():
        df = remove_duplicates(df)
        actions_taken['Remove Duplicates'] = True

    categorical_columns = df.select_dtypes(include=['object']).columns
    if any(df[col].nunique() < 20 and col not in ['name', 'author', 'narrator'] for col in categorical_columns):
        df, mappings = encode_categorical_columns(df)
        actions_taken['Encode Categorical Columns'] = True

    df = split_caps_columns(df)
    actions_taken['Split Caps Columns'] = True

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df = detect_and_remove_outliers(df)
        actions_taken['Detect And Remove Outliers'] = True

    uniform_prefixes = find_uniform_prefixes(df)
    if uniform_prefixes:
        df = clean_uniform_prefixes(df, uniform_prefixes)
        actions_taken['Clean Uniform Prefixes'] = uniform_prefixes

    uniform_postfixes = find_uniform_postfixes(df)
    if uniform_postfixes:
        df = clean_uniform_postfixes(df, uniform_postfixes)
        actions_taken['Clean Uniform Postfixes'] = uniform_postfixes

    uniform_substrings = find_uniform_substrings(df)
    if uniform_substrings:
        df = clean_uniform_substrings(df, uniform_substrings)
        actions_taken['Clean Uniform Substrings'] = uniform_substrings

    df = remove_highly_missing_columns(df)
    actions_taken['Remove Highly Missing Columns'] = True

    df = standardize_dates(df)
    actions_taken['Standardize Dates'] = True

    return df, actions_taken

def format_for_ml(df, target_column, threshold_missing=0.5):
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])
    
    missing_fraction = df.isnull().mean()
    columns_to_remove = missing_fraction[missing_fraction > threshold_missing].index
    df = df.drop(columns=columns_to_remove)
    
    columns_to_remove = []
    irrelevant_keywords = ['id', 'name', 'identifier', 'code', 'language', 'zip', 'address', 'phone', 'email', 'city', 'state', 'country']
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains(r'^\d+$').any():
            columns_to_remove.append(col)
        elif any(keyword in col.lower() for keyword in irrelevant_keywords):
            columns_to_remove.append(col)
    df = df.drop(columns=columns_to_remove)
    
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    X = pd.get_dummies(X)
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        logging.debug("Received file upload request")
        if 'file' not in request.files:
            logging.error("No file part in the request")
            return jsonify({'message': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            logging.error("No selected file")
            return jsonify({'message': 'No selected file'}), 400

        options = json.loads(request.form.get('options', '[]'))
        anonymize = json.loads(request.form.get('anonymize', 'false'))
        anonymize_columns = request.form.get('anonymize_columns', '')
        ml_format = json.loads(request.form.get('ml_format', 'false'))
        target_column = request.form.get('target_column', '')

        logging.debug(f"Received options: {options}")
        logging.debug(f"Anonymize: {anonymize}, Anonymize Columns: {anonymize_columns}")
        logging.debug(f"ML Format: {ml_format}, Target Column: {target_column}")

        logging.debug(f"Received file: {file.filename}")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        logging.debug(f"DataFrame head: \n{df.head()}")

        if 'Lower Case Columns' in options:
            df = lower_case_columns(df)
        if 'Remove Duplicates' in options:
            df = remove_duplicates(df)
        if 'Encode Categorical Columns' in options:
            df, mappings = encode_categorical_columns(df)
            save_mappings_to_csv(mappings, 'audible_mappings.csv')
        if 'Fix Missing Values' in options:
            df = fix_missing_values(df)
        if 'Clean Uniform Prefixes' in options:
            uniform_prefixes = find_uniform_prefixes(df)
            df = clean_uniform_prefixes(df, uniform_prefixes)
        if 'Clean Uniform Postfixes' in options:
            uniform_postfixes = find_uniform_postfixes(df)
            df = clean_uniform_postfixes(df, uniform_postfixes)
        if 'Clean Uniform Substrings' in options:
            uniform_substrings = find_uniform_substrings(df)
            df = clean_uniform_substrings(df, uniform_substrings)
        
        if anonymize and anonymize_columns:
            columns_to_anonymize = [col.strip() for col in anonymize_columns.split(',')]
            df = anonymize_columns(df, columns_to_anonymize)
        
        if ml_format and target_column:
            X, y = format_for_ml(df, target_column)
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + filename)
            save_to_csv(X, cleaned_filepath)
        else:
            cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + filename)
            save_to_csv(df, cleaned_filepath)

        return jsonify({'message': 'File uploaded and read successfully', 'cleaned_file': cleaned_filepath})
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

@app.route('/static/files/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
