import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load DataFrame
fp = 'test-data/audible/audible_uncleaned.csv'
df = pd.read_csv(fp)

# Basic cleaning functions

# Turning the columns to lower case
def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Finding missing values
def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

# Fixing missing values
def fix_missing_values(df):
    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    
    # Impute missing values for numeric columns if they exist
    if not numeric_cols.empty:
        numeric_imputer = SimpleImputer(strategy='mean')
        df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    
    # Impute missing values for non-numeric columns if they exist
    if not non_numeric_cols.empty:
        non_numeric_imputer = SimpleImputer(strategy='most_frequent')
        df[non_numeric_cols] = non_numeric_imputer.fit_transform(df[non_numeric_cols])
    
    return df

# Remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()

# Label encoding
def label_encode(df, columns, mappings):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        mappings[col] = {index: label for index, label in enumerate(le.classes_)}
    return df, mappings

# Save to CSV
def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

# Save mappings to CSV
def save_mappings_to_csv(mappings, filename):
    all_mappings = []
    for col, mapping in mappings.items():
        for index, label in mapping.items():
            all_mappings.append({'column': col, 'numeric_value': index, 'category': label})
    mappings_df = pd.DataFrame(all_mappings)
    mappings_df.to_csv(filename, index=False)

# Function to encode categorical columns with fewer than 20 unique categories
def encode_categorical_columns(df, threshold=20):
    categorical_columns = df.select_dtypes(include=['object']).columns
    mappings = {}
    for col in categorical_columns:
        if df[col].nunique() < threshold:
            df, mappings = label_encode(df, [col], mappings)
    return df, mappings

# Function to find uniform prefixes
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

# Function to clean uniform prefixes
def clean_uniform_prefixes(df, prefixes):
    for col, prefix in prefixes.items():
        pattern = re.compile(rf'^{re.escape(prefix)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Function to find uniform postfixes
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

# Function to clean uniform postfixes
def clean_uniform_postfixes(df, postfixes):
    for col, postfix in postfixes.items():
        pattern = re.compile(rf'{re.escape(postfix)}$', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Function to find uniform substrings
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

# Function to clean uniform substrings
def clean_uniform_substrings(df, substrings):
    for col, substring in substrings.items():
        pattern = re.compile(rf'{re.escape(substring)}', re.IGNORECASE)
        df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x)) if isinstance(x, str) else x)
    return df

# Apply the cleaning pipeline
df = lower_case_columns(df)
df = remove_duplicates(df)
df, mappings = encode_categorical_columns(df)  # Encode columns before fixing missing values
df = fix_missing_values(df)

# Identify and clean uniform prefixes, postfixes, and substrings
uniform_prefixes = find_uniform_prefixes(df)
df = clean_uniform_prefixes(df, uniform_prefixes)
uniform_postfixes = find_uniform_postfixes(df)
df = clean_uniform_postfixes(df, uniform_postfixes)
uniform_substrings = find_uniform_substrings(df)
df = clean_uniform_substrings(df, uniform_substrings)

# Save the cleaned DataFrame
save_to_csv(df, 'cleaned_audible.csv')
# Save the mappings DataFrame
save_mappings_to_csv(mappings, 'audible_mappings.csv')

print("Cleaned DataFrame:\n", df.head())
print("Mappings:\n", mappings)
print("Uniform Prefixes Cleaned:\n", uniform_prefixes)
print("Uniform Postfixes Cleaned:\n", uniform_postfixes)
print("Uniform Substrings Cleaned:\n", uniform_substrings)
