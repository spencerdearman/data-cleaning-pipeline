import pandas as pd
import numpy as np
import os
import json
import logging
import re
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
from flask_cors import CORS
from hashlib import sha256

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Convert all column names to lower case
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Find the number of missing values in each column
def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

# Fix missing values by imputing mean for numeric columns and most frequent value for non-numeric columns
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

# Remove duplicate rows from the dataframe
def remove_duplicates(df):
    return df.drop_duplicates()

# Label encode specified columns
def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Save the dataframe to a CSV file
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Encode categorical columns with fewer than a threshold number of unique values
def encode_categorical_columns(df, threshold=20):
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].nunique() < threshold:
            df = label_encode(df, [col])
    return df

# Find uniform prefixes in string columns
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

# Remove uniform prefixes from string columns
def clean_uniform_prefixes(df, prefixes):
    for col, prefix in prefixes.items():
        pattern = re.compile(rf'^{re.escape(prefix)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Find uniform postfixes in string columns
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

# Remove uniform postfixes from string columns
def clean_uniform_postfixes(df, postfixes):
    for col, postfix in postfixes.items():
        pattern = re.compile(rf'{re.escape(postfix)}$', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Find uniform substrings in string columns
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

# Remove uniform substrings from string columns
def clean_uniform_substrings(df, substrings):
    for col, substring in substrings.items():
        pattern = re.compile(rf'{re.escape(substring)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Split column names and string values based on camel case pattern
def split_caps_columns(df):
    def split_caps(text):
        # Split only on the pattern 'CapsCaps' without spaces in between
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    for col in df.columns:
        # Apply the split only if no cell in the column contains spaces and matches the 'CapsCaps' pattern
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

# Detect and remove outliers in numeric columns based on Z-scores
def detect_and_remove_outliers(df, threshold=3):
    numeric_cols = df.select_dtypes(include=[np.number])
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std())
    df_clean = df[(z_scores < threshold).all(axis=1)]
    return df_clean

# Standardize date columns to a uniform format
def standardize_dates(df):
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    common_date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d', 
                           '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y', '%m/%d/%y', '%d/%m/%y',
                           '%d-%b-%Y', '%b-%d-%Y']

    for col in date_columns:
        for fmt in common_date_formats:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt, errors='raise')
                break
            except ValueError:
                continue

        # Convert to a uniform format after parsing
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    return df

# Remove columns with a high percentage of missing values
def remove_highly_missing_columns(df, threshold=0.5):
    missing_fraction = df.isnull().mean()
    columns_to_remove = missing_fraction[missing_fraction > threshold].index
    df = df.drop(columns=columns_to_remove)
    return df

# Anonymize specified columns by hashing their values
def anonymize_columns(df, columns_to_anonymize):
    for col in columns_to_anonymize:
        df[col] = df[col].apply(lambda x: sha256(x.encode()).hexdigest() if isinstance(x, str) else x)
    return df

@app.route('/upload', methods=['POST'])

# Handle file upload and apply data cleaning options
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'message': 'No file part in the request'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'message': 'No selected file'}), 400

        options = json.loads(request.form.get('options', '[]'))

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)

        if 'lower_case_columns' in options:
            df = lower_case_columns(df)
        if 'remove_duplicates' in options:
            df = remove_duplicates(df)
        if 'encode_categorical_columns' in options:
            df = encode_categorical_columns(df)
        if 'fix_missing_values' in options:
            df = fix_missing_values(df)
        if 'clean_uniform_prefixes' in options:
            uniform_prefixes = find_uniform_prefixes(df)
            df = clean_uniform_prefixes(df, uniform_prefixes)
        if 'clean_uniform_postfixes' in options:
            uniform_postfixes = find_uniform_postfixes(df)
            df = clean_uniform_postfixes(df, uniform_postfixes)
        if 'clean_uniform_substrings' in options:
            uniform_substrings = find_uniform_substrings(df)
            df = clean_uniform_substrings(df, uniform_substrings)
        if 'split_caps_columns' in options:
            df = split_caps_columns(df)
        if 'detect_and_remove_outliers' in options:
            df = detect_and_remove_outliers(df)
        if 'standardize_dates' in options:
            df = standardize_dates(df)
        if 'remove_highly_missing_columns' in options:
            df = remove_highly_missing_columns(df)

        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + filename)
        save_to_csv(df, cleaned_filepath)

        return jsonify({'message': 'File uploaded and read successfully', 'cleaned_file': cleaned_filepath})
    except Exception as e:
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

@app.route('/static/files/<filename>')

# Serve a file from the static files directory
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
