import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the saved model
models = joblib.load('./decision_tree_models.joblib',mmap_mode=None)

# Load the dataset to reconstruct encoders
dataset_path = "C:/Users/saich/Downloads/Cough/cough_dataset.csv"
data = pd.read_csv(dataset_path)

# Normalize dataset values (lowercase for consistency)
data = data.apply(lambda col: col.map(lambda x: str(x).strip().lower() if isinstance(x, str) else x))

# Recreate encoders from the normalized dataset
X = data[['Answer1', 'Answer2', 'Answer3', 'Answer4']]
y = data[['Diagnosis', 'Over the counter medication', 'Advice', 'Red flag symptoms', 'Precautions']]

# Create input encoders
input_encoders = {col: LabelEncoder().fit(X[col].astype(str)) for col in X.columns}

# Function to normalize user inputs
def normalize_input(input_value):
    return str(input_value).strip().lower()

# Function to validate the model
def validate_model():
    # print("\nExpected Input Values:")
    # for col, encoder in input_encoders.items():
    #     print(f"{col}: {list(encoder.classes_)}")

    print("\nEnter your answers for the following questions:")
    try:
        # Collect and normalize inputs
        inputs = {
            col: normalize_input(input(f"{col} must be {list(encoder.classes_)} \nEnter: ")) for col,encoder in input_encoders.items()
        }
        # Validate inputs against the encoder classes
        for col, value in inputs.items():
            if value not in input_encoders[col].classes_:
                raise ValueError(f"{col} must be one of: {list(input_encoders[col].classes_)}")

        # Encode the inputs
        input_data = pd.DataFrame([{
            col: input_encoders[col].transform([value])[0]
            for col, value in inputs.items()
        }]) # we encode because to convert categorical to numbers ,transform method will do this,This ensures that the input 
        #data is in the proper format for the model to process.

        # HERE Matching the row in the dataset for exact expected output
        matched_row = data[
            (data['Answer1'] == inputs['Answer1']) &
            (data['Answer2'] == inputs['Answer2']) &
            (data['Answer3'] == inputs['Answer3']) &
            (data['Answer4'] == inputs['Answer4']) 
        ]
#if matching row is not empty then
        if not matched_row.empty: #if matching row is not empty then 
            # Use dataset row as the final output
            output = matched_row.iloc[0][['Diagnosis', 'Over the counter medication', 'Advice', 'Red flag symptoms', 'Precautions']]
            print("\nMapped Output from Dataset:")
            for key, value in output.items():
                print(f"{key}: {value}")
        else: #means matching row is empty
            # Make predictions for each target column
            predictions = {}
            for col, model_info in models.items():
                # Predict using the specific model
                y_pred_encoded = model_info['model'].predict(input_data)
                # to make a prediction based on the input data (that i encoded earlier).
    # Decode prediction,since i encode the inputs with the help of transform now i am using inverse_transform for decode
                predictions[col] = model_info['label_encoder'].inverse_transform(y_pred_encoded)[0]

    # why i need decode?
    #understood

            print("\nPredicted Outputs:")
            for key, value in predictions.items():
                print(f"{key}: {value}")

    except ValueError as e:
        print(f"\nError: {e}")
        print("Please ensure your inputs match the expected values and try again.")

# Run the validation function
if __name__ == "__main__":
    validate_model()