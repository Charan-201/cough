import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# Load the dataset
dataset_path = "C:/Users/saich/Downloads/Cough/cough_dataset.csv"
data = pd.read_csv(dataset_path).copy()

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

# Perform train-test split
X_train, X_test, y_train, y_test = {}, {}, {}, {}

# Split for each target column
for col in target_columns:
    # Encode target column
    le = LabelEncoder()
    y_encoded = le.fit_transform(y[col])
    
    # Split data
    X_train[col], X_test[col], y_train[col], y_test[col] = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Train individual models
    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42
        )
    )
    
    # Fit the model
    model.fit(X_train[col], y_train[col])
    models[col] = {
        'model': model,
        'label_encoder': le
    }

# Evaluation
print("\nModel Performance:")
for col in target_columns:
    # Predict using the model for this column
    y_pred = models[col]['model'].predict(X_test[col])
    
    # Decode predictions back to original labels
    y_pred_decoded = models[col]['label_encoder'].inverse_transform(y_pred)
    y_test_decoded = models[col]['label_encoder'].inverse_transform(y_test[col])
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test[col], y_pred)
    print(f"\n{col}:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))

# Save the models
joblib.dump(models, './cough_diagnosis_models.joblib')
print("\nModels saved successfully.")

# Prediction function
def predict_cough_diagnosis(input_data, models):
    # Prepare input data
    X_input = pd.DataFrame([input_data], columns="C:/Users/saich/Downloads/cough_dataset_Finalised.csv")
    
    # Encode input features
    le = LabelEncoder()
    for col in X_input.columns:
        X_input[col] = le.fit_transform(X_input[col].astype(str))
    
    # Predict for each column
    predictions = {}
    for col, model_info in models.items():
        # Predict using the specific model
        y_pred = model_info['model'].predict(X_input)
        # Decode prediction
        predictions[col] = model_info['label_encoder'].inverse_transform(y_pred)[0]
    
    return predictions

# Example usage
# sample_input = ['Less than a week', 'Productive', 'Yellowish', 'Yes']
# result = predict_cough_diagnosis(sample_input, models)
# print(result)