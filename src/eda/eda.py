import ipaddress
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FraudDataProcessor:
    def __init__(self, creditcard_path, fraud_data_path, ip_data_path):
        self.creditcard_df = creditcard_path
        self.fraud_data_df = fraud_data_path
        self.ip_data_df = ip_data_path

    def show_missing_values(self):
        """Show the number and percentage of missing values for each dataset."""
        datasets = {
            "CreditCard Data": self.creditcard_df,
            "Fraud Data": self.fraud_data_df,
            "IP Address Data": self.ip_data_df
        }
        for name, df in datasets.items():
            print(f"\nMissing values in {name}:")
            missing_values = df.isnull().sum()
            missing_percent = (missing_values / len(df)) * 100
            missing_data = pd.DataFrame({"Missing Values": missing_values, "Percentage": missing_percent})
            print(missing_data[missing_data["Missing Values"] > 0])

    def handle_missing_values(self, method='drop'):
        """Handle missing values by either dropping or imputing."""
        if method == 'drop':
            self.creditcard_df.dropna(inplace=True)
            self.fraud_data_df.dropna(inplace=True)
            self.ip_data_df.dropna(inplace=True)
        elif method == 'impute':
            # Simple imputation, you can replace with more advanced methods if needed
            self.creditcard_df.fillna(self.creditcard_df.median(), inplace=True)
            self.fraud_data_df.fillna(self.fraud_data_df.median(), inplace=True)
            self.ip_data_df.fillna(self.ip_data_df.median(), inplace=True)

    def clean_data(self):
        """Remove duplicates and correct data types."""
        self.creditcard_df.drop_duplicates(inplace=True)
        self.fraud_data_df.drop_duplicates(inplace=True)
        self.ip_data_df.drop_duplicates(inplace=True)

        # Correct data types (if any column type needs changing, do it here)
        # Example: convert IP address columns to string
        if 'ip_address' in self.fraud_data_df.columns:
            self.fraud_data_df['ip_address'] = self.fraud_data_df['ip_address'].astype(str)
        if 'ip_address' in self.ip_data_df.columns:
            self.ip_data_df['ip_address'] = self.ip_data_df['ip_address'].astype(str)

    def univariate_analysis(self):
        """
        Perform univariate analysis for all datasets.
        This includes summary statistics for numerical features and value counts for categorical features.
        """
        print("\nUnivariate Analysis - Summary Statistics")

        # CreditCard Data Analysis
        print("\nCreditCard Data:")
        numeric_columns = self.creditcard_df.select_dtypes(include=['float64', 'int64']).columns
        print(f"Numeric Columns:\n{self.creditcard_df[numeric_columns].describe()}")
        
        # Fraud Data Analysis
        print("\nFraud Data:")
        # Summary statistics for numeric columns
        numeric_columns = self.fraud_data_df.select_dtypes(include=['float64', 'int64']).columns
        print(f"Numeric Columns:\n{self.fraud_data_df[numeric_columns].describe()}")
        
        # Value counts for categorical columns
        categorical_columns = self.fraud_data_df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            print(f"\nValue Counts for {col}:")
            print(self.fraud_data_df[col].value_counts())
        
        # IP Address Data Analysis
        print("\nIP Address Data:")
        print(self.ip_data_df.describe())

    def bivariate_analysis(self):
        """
        Perform bivariate analysis between features.
        This includes a correlation matrix for numerical features and pairwise plots for potential fraud-related variables.
        """
        print("\nBivariate Analysis")
        
        # Example: Correlation matrix for Fraud Data (numerical columns)
        print("Correlation Matrix for Fraud Data:")
        numeric_columns = self.fraud_data_df.select_dtypes(include=['float64', 'int64']).columns
        corr_matrix = self.fraud_data_df[numeric_columns].corr()
        print(corr_matrix)
        
        # Additional visual or numerical checks
        # Example: Relationship between purchase_value and other factors
        if 'purchase_value' in self.fraud_data_df.columns and 'age' in self.fraud_data_df.columns:
            purchase_age_corr = self.fraud_data_df['purchase_value'].corr(self.fraud_data_df['age'])
            print(f"\nCorrelation between Purchase Value and Age: {purchase_age_corr}")
        
        # Example: Cross-tab for categorical columns (fraud class vs source/browser)
        if 'class' in self.fraud_data_df.columns:
            print("\nCross-tab of Fraud Class vs Source:")
            print(pd.crosstab(self.fraud_data_df['class'], self.fraud_data_df['source']))
            
            print("\nCross-tab of Fraud Class vs Browser:")
            print(pd.crosstab(self.fraud_data_df['class'], self.fraud_data_df['browser']))


    def convert_ip_to_integer(self, ip_address):
        """Convert IPv4 address to integer format."""
        try:
            return int(ipaddress.IPv4Address(ip_address))
        except ValueError:
            return None

    def merge_datasets_for_geolocation(self, batch_size=10000):
        # Convert IP addresses to integer format
        print("Converting IP addresses to integer format in Fraud_Data.csv...")
        self.fraud_data_df['ip_address_int'] = self.fraud_data_df['ip_address'].apply(self.convert_ip_to_integer)

        print("Converting lower and upper IP bounds to integer format in IpAddress_to_Country.csv...")
        self.ip_data_df['lower_bound_ip_int'] = self.ip_data_df['lower_bound_ip_address'].apply(self.convert_ip_to_integer)
        self.ip_data_df['upper_bound_ip_int'] = self.ip_data_df['upper_bound_ip_address'].apply(self.convert_ip_to_integer)

        # Ensure the columns are numeric
        self.ip_data_df['lower_bound_ip_int'] = pd.to_numeric(self.ip_data_df['lower_bound_ip_int'], errors='coerce')
        self.ip_data_df['upper_bound_ip_int'] = pd.to_numeric(self.ip_data_df['upper_bound_ip_int'], errors='coerce')

        # Check for any null values after conversion and handle them
        if self.ip_data_df[['lower_bound_ip_int', 'upper_bound_ip_int']].isnull().any().any():
            print("Warning: Null values detected in IP address columns after conversion.")
            # Option 1: Drop rows with null IP addresses
            self.ip_data_df = self.ip_data_df.dropna(subset=['lower_bound_ip_int', 'upper_bound_ip_int'])
            # Option 2: Fill NaN with a default value, like 0
            # self.ip_data_df = self.ip_data_df.fillna({'lower_bound_ip_int': 0, 'upper_bound_ip_int': 0})

        # Create an interval index
        print("Creating IntervalIndex for IP ranges...")
        ip_intervals = pd.IntervalIndex.from_arrays(self.ip_data_df['lower_bound_ip_int'], 
                                                    self.ip_data_df['upper_bound_ip_int'], 
                                                    closed='both')

        # Process Fraud_Data in batches
        merged_data = []
        for start in range(0, len(self.fraud_data_df), batch_size):
            end = min(start + batch_size, len(self.fraud_data_df))
            fraud_batch = self.fraud_data_df.iloc[start:end]

            # Find the matching country for each IP address in the batch
            fraud_batch['country'] = fraud_batch['ip_address_int'].apply(lambda x: self._find_country(x, ip_intervals))
            merged_data.append(fraud_batch)

        merged_df = pd.concat(merged_data, ignore_index=True)
        return merged_df

    def _find_country(self, ip_int, ip_intervals):
        """Find country for a given IP address based on the interval index."""
        if pd.isna(ip_int):
            return None
        # Check if the IP falls within any range
        matching_index = ip_intervals.get_loc(ip_int, method='nearest')
        if matching_index is not None:
            return self.ip_data_df.iloc[matching_index]['country']
        return None
    def feature_engineering(self, df):
        """Feature engineering for Fraud Data."""
        
        # Create a proxy transaction_id based on user_id and purchase_time
        df['transaction_id'] = df.groupby(['user_id', 'purchase_time']).ngroup()
        
        # Transaction frequency (count of transactions per user)
        df['transaction_count'] = df.groupby('user_id')['transaction_id'].transform('count')
        
        # Transaction velocity (number of transactions divided by the time between them)
        df['transaction_velocity'] = df.groupby('user_id')['purchase_time'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().abs()
        
        # Handle cases where there's only one transaction (avoid division by zero)
        df['transaction_velocity'] = df['transaction_count'] / (df['transaction_velocity'] + 1)
        
        # Time-based features
        df['signup_to_purchase'] = (pd.to_datetime(df['purchase_time']) - pd.to_datetime(df['signup_time'])).dt.total_seconds()
        
        # Return the processed dataframe
        return df


    def normalize_and_scale(self, df):
        """Normalize and scale numerical features."""
        scaler = StandardScaler()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        print("\nData after Normalization and Scaling:")
        print(df[numerical_cols].head())

        return df

    def encode_categorical_features(self, df):
        """Encode categorical features."""
        label_enc = LabelEncoder()
        categorical_cols = df.select_dtypes(include=[object]).columns
        for col in categorical_cols:
            df[col] = label_enc.fit_transform(df[col])

        print("\nData after Encoding Categorical Features:")
        print(df[categorical_cols].head())

        return df
