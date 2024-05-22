import pandas as pd
import numpy as np
import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from sklearn.preprocessing import LabelEncoder
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
from flask_cors import CORS
from hashlib import sha256

# Create a Flask app
app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'secretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Converting all column names to lower case
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Finding the number of missing values in each column
def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

# Fixing missing values by imputing mean for numeric columns and most frequent value for non-numeric columns
def fix_missing_values(df):
    # Finding numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns

    # If the numeric columns are not empty, impute the missing values
    if not numeric_cols.empty:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])

    # If the non-numeric columns are not empty, impute the missing values
    if not non_numeric_cols.empty:
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')
        df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])

    return df

# Removing duplicate rows from the dataframe
def remove_duplicates(df):
    return df.drop_duplicates()

# Label encoding specified columns
def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        
    return df

# Encoding categorical columns with that have less than 20 unique values
def encode_categorical_columns(df, threshold=20):
    # Selecting only the categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if df[col].nunique() < threshold:
            df = label_encode(df, [col])

    return df

# Finding uniform prefixes in string columns
def find_uniform_prefixes(df):
    prefixes = {}

    # Extracting the prefix from the first non-null value in each column
    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)

        # If the column has non-null values
        if not col_values.empty:
            prefix = col_values.str.extract(r'^([^:]+:)', expand=False).mode()
            if not prefix.empty:
                prefix = prefix[0]

                # If the prefix is found in all values of the column
                if all(col_values.str.startswith(prefix)):
                    prefixes[col] = prefix

    return prefixes

# Removing uniform prefixes from string columns
def clean_uniform_prefixes(df, prefixes):
    for col, prefix in prefixes.items():
        # Removing the prefix from each value in the column
        pattern = re.compile(rf'^{re.escape(prefix)}', re.IGNORECASE)

        # Removing the prefix from each value in the column
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)

    return df

# Finding uniform postfixes in string columns
def find_uniform_postfixes(df):
    postfixes = {}

    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)
        if not col_values.empty:
            # Extracting the postfix from the last non-null value in each column
            postfix = col_values.str.extract(r'([^:]+:)$', expand=False).mode()
            if not postfix.empty:
                postfix = postfix[0]

                # If the postfix is found in all values of the column
                if all(col_values.str.endswith(postfix)):
                    postfixes[col] = postfix

    return postfixes

# Removing uniform postfixes from string columns
def clean_uniform_postfixes(df, postfixes):
    for col, postfix in postfixes.items():
        # Creating the regex pattern to match the postfix
        pattern = re.compile(rf'{re.escape(postfix)}$', re.IGNORECASE)

        # Removing the postfix from each value in the column
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)

    return df

# Finding uniform substrings in string columns
def find_uniform_substrings(df):
    substrings = {}
    for col in df.select_dtypes(include=['object']).columns:
        col_values = df[col].dropna().astype(str)
        if not col_values.empty:
            substring = col_values.mode()
            if not substring.empty:
                substring = substring[0]

                # If the substring is found in all values of the column
                if all(col_values == substring):
                    substrings[col] = substring

    return substrings

# Removing uniform substrings from string columns
def clean_uniform_substrings(df, substrings):
    for col, substring in substrings.items():
        # Creating the regex pattern to match the substring
        pattern = re.compile(rf'{re.escape(substring)}', re.IGNORECASE)

        # Removing the substring from each value in the column
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)

    return df

# Splitting column names and string values based on camel case pattern
def split_caps_columns(df):
    def split_caps(text):
        # Splitting only on the pattern with 2 capitals without a space in between
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    for col in df.columns:
        # Applying the split only if no cell in the column contains spaces and matches the pattern
        if df[col].apply(lambda x: isinstance(x, str) and ' ' in x).any():
            continue

        # Finding the camel case pattern in the column
        mask = df[col].apply(lambda x: isinstance(x, str) and bool(re.search(r'[a-z][A-Z]', x)))

        # If the pattern is found in the column
        if mask.any():
            # Splitting the column into two new columns based on the pattern
            split_series = df.loc[mask, col].apply(split_caps)
            new_cols = split_series.str.split(' ', expand=True, n=1)
            df[f'{col} first'] = new_cols[0]
            df[f'{col} last'] = new_cols[1]
            df = df.drop(columns=[col])

    return df

# Detecting and remove outliers in numeric columns based on Z-scores
def detect_and_remove_outliers(df, threshold=3):
    # Selecting only the numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])

    # Calculating the Z-scores for each numeric column
    z_scores = np.abs((numeric_cols - numeric_cols.mean()) / numeric_cols.std())

    # Removing the rows with Z-scores above threshold
    df_clean = df[(z_scores < threshold).all(axis=1)]

    return df_clean

# Standardizing date columns to a uniform format
def standardize_dates(df):
    date_columns = [col for col in df.columns if 'date' in col.lower()]

    # Common date formats
    common_date_formats = ['%d-%m-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y.%m.%d', 
                           '%d-%m-%y', '%m-%d-%Y', '%m-%d-%y', '%m/%d/%y', '%d/%m/%y',
                           '%d-%b-%Y', '%b-%d-%Y']

    # Parsing each date column with the common date formats
    for col in date_columns:
        for fmt in common_date_formats:
            try:
                df[col] = pd.to_datetime(df[col], format=fmt, errors='raise')
                break
            except ValueError:
                continue

        # Converting to a uniform format after parsing
        df[col] = df[col].dt.strftime('%Y-%m-%d')

    return df

# Removing columns with a high percentage of missing values
def remove_highly_missing_columns(df, threshold=0.5):
    missing_fraction = df.isnull().mean()

    # Removing columns with missing fraction above threshold
    columns_to_remove = missing_fraction[missing_fraction > threshold].index
    df = df.drop(columns=columns_to_remove)

    return df

# Anonymizing specified columns by hashing their values
def anonymize_columns(df):
    keywords = ['name', 'key', 'identifier', 'email', 'phone', 'zip', 
                'postal', 'code', 'security', 'id', 'ip', 'ssn']
    for col in df.columns:
        if any(keyword in col.lower() for keyword in keywords):
            df[col] = df[col].apply(lambda x: sha256(x.encode()).hexdigest() if isinstance(x, str) else x)

    return df

# Saving the dataframe to a CSV file
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

# IF I CONTINUE THIS PROJECT, HERE IS WHERE I WANT TO ADD THE ML FORMATTER FUNCTION

@app.route('/upload', methods=['POST'])

# Handling file upload and apply data cleaning options
def upload_file():
    try:
        # Checking if the post request has the file part
        file = request.files['file']
        if file.filename == '':
            # No file selected
            return jsonify({'message': 'No selected file'}), 400

        # Checking if the file is allowed
        options = json.loads(request.form.get('options', '[]'))

        # Saving the file to the upload folder
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Reading the file into a pandas dataframe
        df = pd.read_csv(filepath)

        # Applying the selected data cleaning options
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
        if 'anonymize_columns' in options:
            df = anonymize_columns(df)

        # Saving the cleaned dataframe to a new CSV file
        cleaned_filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'cleaned_' + filename)
        save_to_csv(df, cleaned_filepath)

        return jsonify({'message': 'File uploaded and read successfully', 'cleaned_file': cleaned_filepath})
    except Exception as e:
        return jsonify({'message': f'Error processing file: {str(e)}'}), 500

@app.route('/static/files/<filename>')

# Allowing for downloading the cleaned file
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
