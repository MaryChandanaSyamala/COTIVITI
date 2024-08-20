import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load the CSV file
file_path = 'C:/ProjectC/patients.csv' 
patients_df = pd.read_csv(file_path)

# Convert categorical variables into numerical format
label_encoder = LabelEncoder()
patients_df['Gender'] = label_encoder.fit_transform(patients_df['Gender'])
patients_df['Location'] = label_encoder.fit_transform(patients_df['Location'])
patients_df['SelfAssessedHealthStatus'] = label_encoder.fit_transform(patients_df['SelfAssessedHealthStatus'])

# Define features and target variable
X = patients_df[['Age', 'Diastolic', 'Gender', 'Height', 'Smoker', 'Systolic', 'Weight']]
y = patients_df['SelfAssessedHealthStatus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred_rf = rf_model.predict(X_test_scaled)

# Check the accuracy of the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Model Accuracy: {accuracy_rf * 100:.2f}%")

# Simulate CPG mapping based on model predictions
CPG_mapping = {
    0: "CPG 1: Follow a strict diet and exercise regimen. Monitor blood pressure regularly.",
    1: "CPG 2: Annual health check-ups are recommended. Consider a smoking cessation program.",
    2: "CPG 3: Regular cardiovascular screenings are necessary. Begin medication as needed."
}

# Print unique predicted values to see what's being output by the model
unique_predictions = set(y_pred_rf)
print(f"Unique predictions: {unique_predictions}")

# Ensure all predictions are in the mapping
for pred in unique_predictions:
    if pred not in CPG_mapping:
        CPG_mapping[pred] = "Default CPG: Consult a healthcare provider for further evaluation."

# Generate CPG recommendations for the test set predictions
CPG_recommendations = [CPG_mapping[pred] for pred in y_pred_rf]

# Combine test set data with CPG recommendations for display
recommendations_df = X_test.copy()
recommendations_df['Predicted_Health_Status'] = y_pred_rf
recommendations_df['CPG_Recommendation'] = CPG_recommendations

# Display a few examples
print(recommendations_df.head())