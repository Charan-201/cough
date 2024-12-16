import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
dataset_path = "C:/Users/saich/Downloads/Cough/cough_dataset.csv"
data = pd.read_csv(dataset_path)

# Preprocessing function
def preprocess_data(data):
    # Encode input features
    X = data[['Answer1', 'Answer2', 'Answer3', 'Answer4']].copy()
    
    # Encode categorical features
    le = LabelEncoder()
    for col in X.columns:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Prepare target variables
    y = {}
    target_columns = ['Diagnosis', 'Over the counter medication', 'Advice', 'Red flag symptoms', 'Precautions']
    
    for col in target_columns:
        # Binarize the targets
        y[col] = data[col].astype(str)
    
    return X, y

# Preprocess the data
X, y = preprocess_data(data)

# Prepare models
models = {}
target_columns = list(y.keys())

# Split for each target column and train models
for col in target_columns:
    # Encode target column
    le = LabelEncoder()
    y_encoded = le.fit_transform(y[col].astype(str))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Create and train model
    model = make_pipeline(
        StandardScaler(),
        DecisionTreeClassifier(
            max_depth=10,  # Prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    
    # Store model and encoder
    models[col] = {
        'model': model,
        'label_encoder': le,
        'X_test': X_test,
        'y_test': y_test
    }

# Evaluation
print("\nModel Performance:")
for col in target_columns:
    # Get test data and model info
    model_info = models[col]
    X_test = model_info['X_test']
    y_test = model_info['y_test']
    
    # Make predictions
    y_pred = model_info['model'].predict(X_test)
    
    # Decode predictions and actual values
    y_pred_decoded = model_info['label_encoder'].inverse_transform(y_pred)
    y_test_decoded = model_info['label_encoder'].inverse_transform(y_test)
    
    # Calculate and print metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{col}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

# Save the models
joblib.dump(models, './decision_tree_models.joblib')
print("\nModels saved successfully.")
joblib.load('./decision_tree_models.joblib',mmap_mode=None)

# Example prediction function
def predict_diagnosis(input_data, models):
    """
    Make predictions using the trained models
    
    Args:
        input_data: list of 4 values [Answer1, Answer2, Answer3, Answer4,]
        models: trained models dictionary
    """
    # Prepare input data
    X_input = pd.DataFrame([input_data], columns=['Answer1', 'Answer2', 'Answer3', 'Answer4'])
    
    # Encode input features
    le = LabelEncoder()
    for col in X_input.columns:
        X_input[col] = le.fit_transform(X_input[col].astype(str))
    
    # Make predictions
    predictions = {}
    for col, model_info in models.items():
        # Get prediction
        y_pred = model_info['model'].predict(X_input)
        # Decode prediction
        predictions[col] = model_info['label_encoder'].inverse_transform(y_pred)[0]
    
    return predictions

# Example usage:
# test_input = ['Less than a week', 'Productive', 'Yellowish', 'Yes']
# predictions = predict_diagnosis(test_input, models)
# print("\nPredictions for test input:", predictions)