import joblib
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

class ModelExplainability:
    def __init__(self, model_path, X_train, X_test, feature_names, model_type="tree"):
        """
        Initialize with the model path, training data, test data, and feature names.
        :param model_path: Path to the saved model
        :param X_train: Training data (used to fit SHAP or LIME explainers)
        :param X_test: Test data (used for generating explanations)
        :param feature_names: List of feature names (important for LIME and SHAP plots)
        :param model_type: Type of model ('tree' for tree-based, 'blackbox' for other models)
        """
        self.model_type = model_type
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        """
        Load the saved model based on the type.
        :param model_path: Path to the saved model file
        :return: Loaded model
        """
        if self.model_type in ['tree', 'classical']:
            return joblib.load(model_path)
        elif self.model_type == 'blackbox' or self.model_type == 'deep_learning':
            return load_model(model_path)

    def shap_explain(self):
        """
        Use SHAP to explain model predictions.
        Provides summary, force, and dependence plots.
        """
        # Initialize SHAP explainer based on model type
        if self.model_type == "tree":
            explainer = shap.TreeExplainer(self.model, feature_perturbation='interventional')
        else:
            explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
        
        # Calculate SHAP values for test data
        try:
            shap_values = explainer.shap_values(self.X_test, check_additivity=False)
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return
        
        # SHAP Summary Plot (for global explanation)
        shap.summary_plot(shap_values[1], self.X_test, feature_names=self.feature_names)
        plt.show()

        # SHAP Force Plot (for a single prediction explanation)
        shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], self.X_test.iloc[0, :], matplotlib=True)
        plt.show()

        # SHAP Dependence Plot (for specific feature vs model output)
        shap.dependence_plot(0, shap_values[1], self.X_test, feature_names=self.feature_names)
        plt.show()


    # def shap_explain(self):
    #     """
    #     Use SHAP to explain model predictions.
    #     Provides summary, force, and dependence plots.
    #     """
    #     # Initialize SHAP explainer based on model type
    #     if self.model_type == "tree":
    #         explainer = shap.TreeExplainer(self.model)
    #     else:
    #         explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
        
    #     # Calculate SHAP values for test data
    #     shap_values = explainer.shap_values(self.X_test, check_additivity=False)
        
    #     # SHAP Summary Plot (for global explanation)
    #     shap.summary_plot(shap_values[1], self.X_test, feature_names=self.feature_names)
    #     plt.show()

    #     # SHAP Force Plot (for a single prediction explanation)
    #     shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], self.X_test.iloc[0, :], matplotlib=True)
    #     plt.show()

    #     # SHAP Dependence Plot (for specific feature vs model output)
    #     shap.dependence_plot(0, shap_values[1], self.X_test, feature_names=self.feature_names)
    #     plt.show()

    def lime_explain(self, instance_idx=0):
        """
        Use LIME to explain a single prediction.
        :param instance_idx: Index of the instance to explain (default is 0)
        """
        # Initialize LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values, 
            feature_names=self.feature_names, 
            class_names=["Not Fraud", "Fraud"],
            mode='classification'
        )
        
        # Select an instance from the test set for explanation
        instance = self.X_test.iloc[instance_idx]
        
        # Explain the prediction of the model for the selected instance
        explanation = explainer.explain_instance(instance.values, self.model.predict_proba)
        
        # LIME Feature Importance Plot
        explanation.show_in_notebook(show_table=True)

    def shap_explain_single(self, instance_idx=0):
        """
        SHAP Force Plot for a single instance.
        :param instance_idx: Index of the instance to explain (default is 0)
        """
        # Initialize SHAP explainer
        if self.model_type == "tree":
            explainer = shap.TreeExplainer(self.model, feature_perturbation='interventional')
        else:
            explainer = shap.KernelExplainer(self.model.predict_proba, self.X_train)
        
        # Calculate SHAP values for the specific instance
        shap_values = explainer.shap_values(self.X_test.iloc[[instance_idx]])
        
        # SHAP Force Plot for a single prediction
        shap.force_plot(explainer.expected_value[1], shap_values[1][0, :], self.X_test.iloc[instance_idx, :], matplotlib=True)
        plt.show()

