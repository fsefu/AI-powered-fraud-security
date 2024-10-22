import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, RNN
from tensorflow.keras.optimizers import Adam

class FraudDetectionModel:
    def __init__(self, df, target_column, experiment_name='FraudDetection'):
        """
        Initialize with the dataset and target column.
        """
        self.df = df
        self.target_column = target_column
        self.experiment_name = experiment_name
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "MLP": MLPClassifier()
        }
        mlflow.set_experiment(experiment_name)
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data by separating features and target, and performing train-test split.
        """
        X = self.df.drop(self.target_column, axis=1)
        y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def train_classical_models(self):
        """
        Train classical machine learning models.
        """
        self.results = {}
        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=model_name):
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                roc_auc = roc_auc_score(self.y_test, y_pred)

                # Logging metrics to MLflow
                mlflow.log_param("model_name", model_name)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
                mlflow.log_metric("F1 Score", f1)
                mlflow.log_metric("ROC AUC", roc_auc)

                mlflow.sklearn.log_model(model, model_name)

                # Store results in dictionary
                self.results[model_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1,
                    "ROC AUC": roc_auc
                }

        return self.results

    def create_cnn_model(self, input_shape):
        """
        Create a CNN model.
        """
        model = Sequential()
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_deep_learning_model(self, model_name, model, epochs=10, batch_size=32):
        """
        Train deep learning models like CNN, LSTM.
        """
        # Ensure input shape is compatible with the model (add additional dimension if necessary)
        X_train_reshaped = self.X_train.values.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        X_test_reshaped = self.X_test.values.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        with mlflow.start_run(run_name=model_name):
            history = model.fit(X_train_reshaped, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
            
            loss, accuracy = model.evaluate(X_test_reshaped, self.y_test)
            
            # Logging metrics to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("Loss", loss)
            mlflow.log_metric("Accuracy", accuracy)
            
            # Save model
            mlflow.keras.log_model(model, model_name)

            return {"Loss": loss, "Accuracy": accuracy}

    def create_lstm_model(self, input_shape):
        """
        Create an LSTM model.
        """
        model = Sequential()
        model.add(LSTM(50, activation='relu', input_shape=input_shape))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        return model

