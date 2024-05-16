import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib
import re
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Load DataFrame
fp = 'test-data/jobs/Uncleaned_DS_jobs.csv'
df = pd.read_csv(fp)

# Define all the functions

def lower_case_columns(df):
    df.columns = df.columns.str.lower()
    return df

def find_missing_values(df):
    missing_values = df.isnull().sum()
    return missing_values

def fix_missing_values(df, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    imp_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return imp_df

def fix_missing_values_knn(df, n_neighbors=5):
    numeric_df = df.select_dtypes(include=[np.number])
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imp_numeric_df = pd.DataFrame(imputer.fit_transform(numeric_df), columns=numeric_df.columns)
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    imp_df = pd.concat([imp_numeric_df, non_numeric_df], axis=1)
    return imp_df

def remove_duplicates(df):
    return df.drop_duplicates()

def detect_outliers(df):
    iso_forest = IsolationForest(contamination=0.1)
    numeric_df = df.select_dtypes(include=[np.number])
    outliers = iso_forest.fit_predict(numeric_df)
    return outliers

def handle_outliers(df, outliers):
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df = numeric_df[outliers == 1]
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    df_cleaned = pd.concat([numeric_df, non_numeric_df], axis=1)
    return df_cleaned

def remove_high_missing_columns(df, threshold=0.5):
    missing_percentage = df.isnull().mean()
    return df.loc[:, missing_percentage < threshold]

def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_df = df.select_dtypes(include=[np.number])
    scaled_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    df_scaled = pd.concat([scaled_df, non_numeric_df], axis=1)
    return df_scaled

def standardize_data(df):
    scaler = StandardScaler()
    numeric_df = df.select_dtypes(include=[np.number])
    standardized_df = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
    non_numeric_df = df.select_dtypes(exclude=[np.number])
    df_standardized = pd.concat([standardized_df, non_numeric_df], axis=1)
    return df_standardized

def label_encode(df, columns):
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

def one_hot_encode(df, columns):
    return pd.get_dummies(df, columns=columns)

def anonymize_value(value):
    return hashlib.md5(str(value).encode()).hexdigest()

def anonymize_column(df, column):
    df[column] = df[column].apply(anonymize_value)
    return df

def summary_statistics(df):
    return df.describe()

def save_to_csv(df, filename):
    df.to_csv(filename, index=False)

PII_KEYWORDS = [
    'name', 'email', 'phone', 'address', 'dob', 'ssn', 'social_security', 'credit_card', 'ccv', 'zip', 'postal', 'birthday',
]

def contains_pii_keywords(column_name):
    column_name_lower = column_name.lower()
    for keyword in PII_KEYWORDS:
        if re.search(r'\b' + keyword + r'\b', column_name_lower):
            return True
    return False

def detect_pii_columns(df):
    pii_columns = []
    for column in df.columns:
        if contains_pii_keywords(column):
            pii_columns.append(column)
    return pii_columns

def feature_selection(df, target_column, k=10):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    selected_features = selector.get_support(indices=True)
    return df.columns[selected_features]

def categorize_data(df, n_clusters=3):
    numerical_df = df.select_dtypes(include=[np.number])
    if len(numerical_df) < n_clusters:
        n_clusters = len(numerical_df)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(numerical_df)
    df['category'] = labels
    return df

def detect_skewed_columns(df, skew_threshold=0.75):
    numeric_df = df.select_dtypes(include=[np.number])
    skewed_columns = numeric_df.apply(lambda x: x.skew()).abs()
    return skewed_columns[skewed_columns > skew_threshold].index.tolist()

# Analysis functions
def analyze_dataset(df):
    analysis_report = {}
    missing_values = find_missing_values(df)
    analysis_report['missing_values'] = missing_values[missing_values > 0]
    duplicates = df.duplicated().sum()
    analysis_report['duplicates'] = duplicates
    outliers = detect_outliers(df)
    analysis_report['outliers'] = (outliers == -1).sum()
    skewed_columns = detect_skewed_columns(df)
    analysis_report['skewed_columns'] = skewed_columns
    pii_columns = detect_pii_columns(df)
    analysis_report['pii_columns'] = pii_columns
    return analysis_report

def suggest_cleaning_functions(analysis_report):
    suggestions = []
    if not analysis_report['missing_values'].empty:
        suggestions.append("fix_missing_values or fix_missing_values_knn")
    if analysis_report['duplicates'] > 0:
        suggestions.append("remove_duplicates")
    if analysis_report['outliers'] > 0:
        suggestions.append("handle_outliers")
    if len(analysis_report['skewed_columns']) > 0:
        suggestions.append("normalize_data or standardize_data")
    if len(analysis_report['pii_columns']) > 0:
        suggestions.append("anonymize_column")
    suggestions.append("lower_case_columns")
    suggestions.append("remove_high_missing_columns")
    suggestions.append("label_encode")
    suggestions.append("one_hot_encode")
    return suggestions

def generate_pipeline(df, suggestions):
    for suggestion in suggestions:
        if suggestion == "fix_missing_values or fix_missing_values_knn":
            df = fix_missing_values_knn(df)  # You can choose fix_missing_values or fix_missing_values_knn
        elif suggestion == "remove_duplicates":
            df = remove_duplicates(df)
        elif suggestion == "handle_outliers":
            outliers = detect_outliers(df)
            df = handle_outliers(df, outliers)
        elif suggestion == "normalize_data or standardize_data":
            df = standardize_data(df)  # You can choose normalize_data or standardize_data
        elif suggestion == "anonymize_column":
            pii_columns = detect_pii_columns(df)
            for col in pii_columns:
                df = anonymize_column(df, col)
        elif suggestion == "lower_case_columns":
            df = lower_case_columns(df)
        elif suggestion == "remove_high_missing_columns":
            df = remove_high_missing_columns(df)
        elif suggestion == "label_encode":
            categorical_columns = df.select_dtypes(include=['object']).columns
            df = label_encode(df, categorical_columns)
        elif suggestion == "one_hot_encode":
            categorical_columns = df.select_dtypes(include=['object']).columns
            df = one_hot_encode(df, categorical_columns)
    return df

# -----------------------------Money Formatting -------------------------------------------


def convert_financial_string_to_number(value):
    if isinstance(value, str):
        # Remove any commas for thousands
        value = value.replace(',', '')
        # Find all parts of the financial string (numbers and denominators)
        parts = re.findall(r'(\d+\.?\d*)\s*(billion|million|thousand)', value, flags=re.IGNORECASE)
        total_value = 0
        # Map each part to its numeric value
        for number, unit in parts:
            number = float(number)
            if 'billion' == unit.lower():
                number *= 1e9
            elif 'million' == unit.lower():
                number *= 1e6
            elif 'thousand' == unit.lower():
                number *= 1e3
            total_value += number
        return total_value if total_value != 0 else value
    return value

def clean_money(value):
    # Handle a range with potential annotations
    if isinstance(value, str):
        # Remove any non-numeric characters except dash and letters for abbreviations
        clean_value = re.sub(r'[^\dKkMm.-]', '', value)
        # Find ranges like '137K-171K'
        range_match = re.findall(r'(\d+\.?\d*)([KkMm]?)', clean_value)
        numbers = []

        for number, multiplier in range_match:
            # Convert number to float and adjust by multiplier
            number = float(number)
            if multiplier.upper() == 'K':
                number *= 1000
            elif multiplier.upper() == 'M':
                number *= 1000000
            numbers.append(number)

        if numbers:
            # Return average of the range if it's a range, otherwise the single number
            return sum(numbers) / len(numbers)
    return value

# Identify columns with financial data based on keywords in their names
def identify_money_columns_by_name(df):
    keywords = ['income', 'salary', 'price', 'cost', 'wage', 'earnings', 'revenue']
    money_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in keywords)]
    print(f"Identified money columns: {money_columns}")  # Debugging output
    return money_columns

# Apply the transformation
def format_money_columns_pipeline(df):
    money_columns = identify_money_columns_by_name(df)
    for col in money_columns:
        df[col] = df[col].apply(lambda x: convert_financial_string_to_number(clean_money(x)))
        print(f"Formatted column {col}:\n{df[col].head()}")  # Debugging output
    return df

# Apply the pipeline
df_cleaned = format_money_columns_pipeline(df)
print("Cleaned DataFrame:\n", df_cleaned.head())

# Save the cleaned DataFrame
df_cleaned.to_csv('cleaned_DS_jobs.csv', index=False)
# ------------------------------------------------------------------------
