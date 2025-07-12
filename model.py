import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class HeartDiseaseDetector:
    """
    A comprehensive heart disease detection system with multiple ML models
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.results = {}
        
    def load_data(self, data_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load and prepare the heart disease dataset
        """
        if df is not None:
            self.data = df.copy()
        else:
            # Load from file if path provided
            self.data = pd.read_csv(data_path)
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Target distribution:\n{self.data['HeartDisease'].value_counts()}")
        
        return self.data
    
    def preprocess_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess the data: handle categorical variables, scale features
        """
        df = self.data.copy()
        
        # Separate features and target
        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease']
        
        # Handle categorical variables
        categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        
        for feature in categorical_features:
            if feature in X.columns:
                le = LabelEncoder()
                X[feature] = le.fit_transform(X[feature])
                self.encoders[feature] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X_array = X.values
        y_array = y.values
        
        print("Data preprocessing completed!")
        print(f"Features: {self.feature_names}")
        
        return X_array, y_array
    
    def split_and_scale_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Split data and apply scaling
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train multiple models with hyperparameter tuning
        """
        models_config = {
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(probability=True, random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['rbf', 'linear', 'poly'],
                    'gamma': ['scale', 'auto']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
            }
        }
        
        print("Training models with hyperparameter tuning...")
        
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'], 
                config['params'], 
                cv=5, 
                scoring='f1',
                n_jobs=-1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Store the best model
            self.models[name] = grid_search.best_estimator_
            
            print(f"Best parameters for {name}: {grid_search.best_params_}")
            print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate all trained models
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION RESULTS")
        print("="*60)
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name} Results:")
            print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
            print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
        
        return self.results
    
    def get_best_model(self) -> Tuple[str, Any]:
        """
        Get the best performing model based on F1 score
        """
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"F1 Score: {self.results[best_model_name]['f1_score']:.4f}")
        
        return best_model_name, best_model
    
    def plot_confusion_matrices(self, X_test: np.ndarray, y_test: np.ndarray):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'], ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, model_name: str = None):
        """
        Plot feature importance for tree-based models
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [self.feature_names[i] for i in indices], 
                      rotation=45)
            plt.tight_layout()
            plt.show()
        else:
            print(f"Feature importance not available for {model_name}")
    
    def predict_single(self, patient_data: Dict, model_name: str = None) -> Dict:
        """
        Make prediction for a single patient
        """
        if model_name is None:
            model_name, _ = self.get_best_model()
        
        model = self.models[model_name.title()]
        scaler = self.scalers['main']
        
        # Convert patient data to dataframe
        patient_df = pd.DataFrame([patient_data])
        
        # Apply same preprocessing
        categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
        for feature in categorical_features:
            if feature in patient_df.columns and feature in self.encoders:
                patient_df[feature] = self.encoders[feature].transform(patient_df[feature])
        
        # Scale the features
        patient_scaled = scaler.transform(patient_df[self.feature_names])
        
        # Make prediction
        prediction = model.predict(patient_scaled)[0]
        probability = model.predict_proba(patient_scaled)[0]
        
        result = {
            'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability_no_disease': probability[0],
            'probability_disease': probability[1],
            'confidence': max(probability),
            'model_used': model_name
        }
        
        return result
    
    def save_models(self, filepath_prefix: str = 'heart_disease_model'):
        """
        Save all trained models and preprocessors
        """
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f'{filepath_prefix}_{name.lower()}.pkl')
        
        # Save scalers and encoders
        joblib.dump(self.scalers, f'{filepath_prefix}_scalers.pkl')
        joblib.dump(self.encoders, f'{filepath_prefix}_encoders.pkl')
        joblib.dump(self.feature_names, f'{filepath_prefix}_features.pkl')
        
        print(f"Models and preprocessors saved with prefix: {filepath_prefix}")
    
    def load_model(self, filepath_prefix: str = 'heart_disease_model', name:str=None):
        try:
            if name is None:
                name, _ = self.get_best_model()
                
            self.models[name.title()] = joblib.load(f'{filepath_prefix}_{name}.pkl')
        except FileNotFoundError:
            print(f"Model {name} not found")
        
        self.scalers = joblib.load(f'{filepath_prefix}_scalers.pkl')
        self.encoders = joblib.load(f'{filepath_prefix}_encoders.pkl')
        self.feature_names = joblib.load(f'{filepath_prefix}_features.pkl')
        
        print("Models and preprocessors loaded successfully!")


# Example usage
def main():
    # Initialize detector
    detector = HeartDiseaseDetector()
    
    # Load data
    detector.load_data(data_path='dataset.csv')
    
    # Preprocess data
    X, y = detector.preprocess_data()
    
    # Split and scale data
    X_train, X_test, y_train, y_test = detector.split_and_scale_data(X, y)
    
    # Train models
    detector.train_models(X_train, y_train)
    
    # Evaluate models
    detector.evaluate_models(X_test, y_test)
    
    # Get best model
    best_model_name, best_model = detector.get_best_model()
    
    # Save models
    detector.save_models()
    
    return detector

if __name__ == "__main__":
    detector = main()