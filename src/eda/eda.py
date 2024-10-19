import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FraudDataProcessor:
    def __init__(self, creditcard_path, fraud_data_path, ip_data_path):
        self.creditcard_df = pd.read_csv(creditcard_path)
        self.fraud_data_df = pd.read_csv(fraud_data_path)
        self.ip_data_df = pd.read_csv(ip_data_path)

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
        """Perform univariate analysis for all datasets."""
        print("\nUnivariate Analysis - Summary Statistics")
        print("\nCreditCard Data:")
        print(self.creditcard_df.describe())

        print("\nFraud Data:")
        print(self.fraud_data_df.describe())

        print("\nIP Address Data:")
        print(self.ip_data_df.describe())

    def bivariate_analysis(self):
        """Perform bivariate analysis between features."""
        print("\nBivariate Analysis")
        # Example: Correlation matrix for Fraud Data
        corr_matrix = self.fraud_data_df.corr()
        print("Correlation Matrix for Fraud Data:")
        print(corr_matrix)

    def merge_datasets(self):
        """Convert IP addresses to integer and merge Fraud Data with IP Address to Country data."""
        # Convert IP address to integers
        self.ip_data_df['ip_start_int'] = self.ip_data_df['ip_start'].apply(self.ip_to_int)
        self.ip_data_df['ip_end_int'] = self.ip_data_df['ip_end'].apply(self.ip_to_int)

        # Merge fraud data with IP address country data based on the IP range
        self.fraud_data_df['ip_address_int'] = self.fraud_data_df['ip_address'].apply(self.ip_to_int)
        merged_df = pd.merge(self.fraud_data_df, self.ip_data_df, how='left', left_on='ip_address_int',
                             right_on='ip_start_int')

        print("\nMerged Fraud Data with Geolocation:")
        print(merged_df.head())

        return merged_df

    @staticmethod
    def ip_to_int(ip):
        """Convert IP address to an integer."""
        return int(ip.replace('.', ''))

    def feature_engineering(self, df):
        """Feature engineering for Fraud Data."""
        # Transaction frequency and velocity
        df['transaction_count'] = df.groupby('user_id')['transaction_id'].transform('count')
        df['transaction_velocity'] = df['transaction_count'] / df['transaction_time'].diff().abs()

        # Time-based features
        df['hour_of_day'] = pd.to_datetime(df['transaction_time']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['transaction_time']).dt.dayofweek

        print("\nFeature Engineering - Transaction and Time-based features:")
        print(df[['transaction_count', 'transaction_velocity', 'hour_of_day', 'day_of_week']].head())

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
