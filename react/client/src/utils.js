export const cleaningOptionsMap = {
    'Lower Case Columns': 'lower_case_columns',
    'Remove Duplicates': 'remove_duplicates',
    'Encode Categorical Columns': 'encode_categorical_columns',
    'Fix Missing Values': 'fix_missing_values',
    'Clean Prefixes': 'clean_uniform_prefixes',
    'Clean Postfixes': 'clean_uniform_postfixes',
    'Clean Substrings': 'clean_uniform_substrings',
    'Split Capitalized Columns': 'split_caps_columns',
    'Detect and Remove Outliers': 'detect_and_remove_outliers',
    'Standardize Dates': 'standardize_dates',
    'Remove Missing Columns': 'remove_highly_missing_columns',
    'Anonymize Columns': 'anonymize_columns',
  };
  
  export const cleaningOptions = Object.keys(cleaningOptionsMap);
  