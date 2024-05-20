import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from hashlib import sha256

# Load DataFrame
fp = '../test-data/audible/audible_uncleaned.csv'
df = pd.read_csv(fp)

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

# New generalized functions
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
        # Convert to a uniform format after parsing
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

# Decision-making function
def analyze_and_clean(df):
    # Dictionary to store actions taken
    actions_taken = {}

    df = lower_case_columns(df)
    actions_taken['lower_case_columns'] = True
    
    # Check for missing values
    if df.isnull().values.any():
        df = fix_missing_values(df)
        actions_taken['fix_missing_values'] = True

    # Check for duplicates
    if df.duplicated().any():
        df = remove_duplicates(df)
        actions_taken['remove_duplicates'] = True

    # Check for categorical columns with less than threshold unique values for encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    if any(df[col].nunique() < 20 and col not in ['name', 'author', 'narrator'] for col in categorical_columns):
        df, mappings = encode_categorical_columns(df)
        actions_taken['encode_categorical_columns'] = True

    # Split columns with 'CapsCaps' pattern
    df = split_caps_columns(df)
    actions_taken['split_caps_columns'] = True

    # Detect and remove outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if not numeric_cols.empty:
        df = detect_and_remove_outliers(df)
        actions_taken['detect_and_remove_outliers'] = True

    # Identify and clean uniform prefixes, postfixes, and substrings
    uniform_prefixes = find_uniform_prefixes(df)
    if uniform_prefixes:
        df = clean_uniform_prefixes(df, uniform_prefixes)
        actions_taken['clean_uniform_prefixes'] = uniform_prefixes

    uniform_postfixes = find_uniform_postfixes(df)
    if uniform_postfixes:
        df = clean_uniform_postfixes(df, uniform_postfixes)
        actions_taken['clean_uniform_postfixes'] = uniform_postfixes

    uniform_substrings = find_uniform_substrings(df)
    if uniform_substrings:
        df = clean_uniform_substrings(df, uniform_substrings)
        actions_taken['clean_uniform_substrings'] = uniform_substrings

    # Remove columns with high missing values
    df = remove_highly_missing_columns(df)
    actions_taken['remove_highly_missing_columns'] = True

    # Standardize date formats
    df = standardize_dates(df)
    actions_taken['standardize_dates'] = True

    return df, actions_taken

def format_for_ml(df, target_column, threshold_missing=0.5):
    # Drop columns with a single unique value
    for col in df.columns:
        if df[col].nunique() == 1:
            df = df.drop(columns=[col])
    
    # Drop columns with a high percentage of missing values
    missing_fraction = df.isnull().mean()
    columns_to_remove = missing_fraction[missing_fraction > threshold_missing].index
    df = df.drop(columns=columns_to_remove)
    
    # Drop columns that are unlikely to be useful for ML (e.g., IDs, names)
    columns_to_remove = []
    irrelevant_keywords = ['id', 'name', 'identifier', 'code', 'language', 'zip', 'address', 'phone', 'email', 'city', 'state', 'country']
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains(r'^\d+$').any():
            columns_to_remove.append(col)
        elif any(keyword in col.lower() for keyword in irrelevant_keywords):
            columns_to_remove.append(col)
    df = df.drop(columns=columns_to_remove)
    
    # Separate target and features
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    # Handle categorical features
    X = pd.get_dummies(X)
    
    # Standardize numerical features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y

df, actions_taken = analyze_and_clean(df)
print("Actions Taken:\n", actions_taken)


train = False
anonymize = True

if train:
    target_column = ''  # Replace with the actual target column name
    X, y = format_for_ml(df, target_column)
    print("X:", X.head())

    # Now you can perform train-test split and use the data for ML
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

# Example of anonymizing columns
if anonymize:
    columns_to_anonymize = ['author first', 'author last']  # Replace with actual columns to anonymize
    df = anonymize_columns(df, columns_to_anonymize)

# Save the cleaned DataFrame and mappings
save_to_csv(df, '../test-data/audible/cleaned_audible.csv')
print("Cleaned DataFrame:\n", df.head())
print("Actions Taken:\n", actions_taken)