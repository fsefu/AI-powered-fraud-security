import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten
import mlflow
import mlflow.sklearn
import pickle
import os

# Define the data preprocessing class
class DataPreprocessor:
    def __init__(self, fraud_data_path, creditcard_data_path, ip_country_data_path):
        self.fraud_data_path = fraud_data_path
        self.creditcard_data_path = creditcard_data_path
        self.ip_country_data_path = ip_country_data_path
        self.scaler = StandardScaler()

    def load_data(self):
        fraud_data = pd.read_csv(self.fraud_data_path)
        creditcard_data = pd.read_csv(self.creditcard_data_path)
        ip_country_data = pd.read_csv(self.ip_country_data_path)
        return fraud_data, creditcard_data, ip_country_data

    def preprocess_fraud_data(self, fraud_data):
        # Convert timestamps to datetime
        fraud_data['signup_time'] = pd.to_datetime(fraud_data['signup_time'])
        fraud_data['purchase_time'] = pd.to_datetime(fraud_data['purchase_time'])

        # Calculate time difference as a feature
        fraud_data['time_diff'] = (fraud_data['purchase_time'] - fraud_data['signup_time']).dt.total_seconds()

        # Encoding categorical columns and handling missing values
        fraud_data = pd.get_dummies(fraud_data, columns=['source', 'browser', 'sex'], drop_first=True)

        # Define the expected dummy columns based on previous analysis
        expected_columns = ['purchase_value', 'source_SEO', 'browser_Chrome', 'sex_M', 'age', 'time_diff']

        # Add missing columns with default value 0
        for col in expected_columns:
            if col not in fraud_data.columns:
                fraud_data[col] = 0

        # Select only the required columns
        X = fraud_data[expected_columns]
        y = fraud_data['class']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def preprocess_creditcard_data(self, creditcard_data):
        X = creditcard_data.drop(columns=['Class'])
        y = creditcard_data['Class']
        X = self.scaler.fit_transform(X)
        return train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model trainer class
class FraudModelTrainer:
    def __init__(self):
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP": MLPClassifier()
        }

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        return metrics

    def run_experiments(self, X_train, X_test, y_train, y_test, model_save_dir="saved_models"):
        mlflow.set_experiment("Fraud Detection Model Training")
        os.makedirs(model_save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                metrics = self.train_and_evaluate(model, X_train, X_test, y_train, y_test)
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, model_name)
                print(f"{model_name} Metrics: {metrics}")
                
                # Save the model for API prediction
                model_path = os.path.join(model_save_dir, f"{model_name}.pkl")
                with open(model_path, "wb") as file:
                    pickle.dump(model, file)
                print(f"Model saved to {model_path}")

    def build_cnn(self, input_shape):
        cnn_model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            Dropout(0.5),
            Flatten(),
            Dense(50, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return cnn_model

    def build_lstm(self, input_shape):
        lstm_model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.5),
            LSTM(32),
            Dense(1, activation='sigmoid')
        ])
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return lstm_model

    def train_deep_learning_models(self, X_train, y_train, X_test, y_test, model_type="CNN"):
        if model_type == "CNN":
            model = self.build_cnn(input_shape=(X_train.shape[1], 1))
        elif model_type == "LSTM":
            model = self.build_lstm(input_shape=(X_train.shape[1], 1))
        
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
        loss, accuracy = model.evaluate(X_test, y_test)
        metrics = {"accuracy": accuracy}
        
        mlflow.log_metrics(metrics)
        return model, metrics

# # Usage Example
# def main():
#     fraud_data_path = 'Fraud_Data.csv'
#     creditcard_data_path = 'creditcard.csv'
#     ip_country_data_path = 'IpAddress_to_Country.csv'
    
#     preprocessor = DataPreprocessor(fraud_data_path, creditcard_data_path, ip_country_data_path)
#     trainer = FraudModelTrainer()

#     fraud_data, creditcard_data, ip_country_data = preprocessor.load_data()
#     X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud = preprocessor.preprocess_fraud_data(fraud_data)
#     X_train_cc, X_test_cc, y_train_cc, y_test_cc = preprocessor.preprocess_creditcard_data(creditcard_data)

#     print("Training on Fraud_Data...")
#     trainer.run_experiments(X_train_fraud, X_test_fraud, y_train_fraud, y_test_fraud)

#     print("Training on CreditCard Data...")
#     trainer.run_experiments(X_train_cc, X_test_cc, y_train_cc, y_test_cc)

#     X_train_dl = X_train_fraud.reshape(-1, X_train_fraud.shape[1], 1)
#     X_test_dl = X_test_fraud.reshape(-1, X_test_fraud.shape[1], 1)
#     model, metrics = trainer.train_deep_learning_models(X_train_dl, y_train_fraud, X_test_dl, y_test_fraud, model_type="CNN")
#     print(f"CNN Model Metrics: {metrics}")

# if __name__ == "__main__":
#     main()
