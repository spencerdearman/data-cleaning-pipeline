from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import logging
from werkzeug.utils import secure_filename
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from scipy import stats
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/files'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        if df[col].nunique() < threshold:
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

def remove_outliers(df, z_thresh=3):
    z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
    df = df[(z_scores < z_thresh).all(axis=1)]
    return df

def normalize_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def standardize_data(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def create_features(df):
    # Example: Creating a new feature based on existing ones
    if 'column1' in df.columns and 'column2' in df.columns:
        df['new_feature'] = df['column1'] * df['column2']
    return df

def process_text(df, column):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    def clean_text(text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word is not None]
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)
    
    df[column] = df[column].apply(lambda x: clean_text(x) if isinstance(x, str) else x)
    return df

def balance_data(df, target_column):
    smote = SMOTE()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_res, y_res = smote.fit_resample(X, y)
    df_res = pd.concat([X_res, y_res], axis=1)
    return df_res

def reduce_dimensions(df, n_components=2):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    pca = PCA(n_components=n_components)
    df_pca = pca.fit_transform(df[numeric_cols])
    df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    df = pd.concat([df, df_pca], axis=1).drop(columns=numeric_cols)
    return df

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
        logging.debug(f"Received options: {options}")

        logging.debug(f"Received file: {file.filename}")
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        logging.debug(f"DataFrame head: \n{df.head()}")

        if 'lower_case_columns' in options:
            df = lower_case_columns(df)
        if 'remove_duplicates' in options:
            df = remove_duplicates(df)
        if 'encode_categorical_columns' in options:
            df, mappings = encode_categorical_columns(df)
            save_mappings_to_csv(mappings, 'audible_mappings.csv')
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
        if 'remove_outliers' in options:
            df = remove_outliers(df)
        if 'normalize_data' in options:
            df = normalize_data(df)
        if 'standardize_data' in options:
            df = standardize_data(df)
        if 'create_features' in options:
            df = create_features(df)
        if 'process_text' in options:
            # Adjust column name as needed
            df = process_text(df, column='text_column')
        if 'balance_data' in options:
            # Adjust target column name as needed
            df = balance_data(df, target_column='target')
        if 'reduce_dimensions' in options:
            df = reduce_dimensions(df)

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
