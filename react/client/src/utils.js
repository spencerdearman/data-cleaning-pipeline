export const cleaningOptionsMap = {
    'Lower Case Columns': 'lower_case_columns',
    'Remove Duplicates': 'remove_duplicates',
    'Encode Categorical Columns': 'encode_categorical_columns',
    'Fix Missing Values': 'fix_missing_values',
    'Clean Uniform Prefixes': 'clean_uniform_prefixes',
    'Clean Uniform Postfixes': 'clean_uniform_postfixes',
    'Clean Uniform Substrings': 'clean_uniform_substrings',
    'Split Caps Columns': 'split_caps_columns',
    'Detect and Remove Outliers': 'detect_and_remove_outliers',
    'Standardize Dates': 'standardize_dates',
    'Remove Highly Missing Columns': 'remove_highly_missing_columns'
  };
  
  export const cleaningOptions = Object.keys(cleaningOptionsMap);
  