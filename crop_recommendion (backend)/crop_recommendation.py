import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Get base directory path (same folder as this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_and_preprocess_data(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: '{file_path}' not found in '{os.getcwd()}'")
        df = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
    except FileNotFoundError as e:
        print(e)
        return None, None
    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return None, None

    numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    df = handle_outliers(df, numerical_features)
    return df, numerical_features

def handle_outliers(df, numerical_features):
    df_cleaned = df.copy()
    for col in numerical_features:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
    return df_cleaned

def prepare_data_for_model(df, numerical_features):
    df_prepared = df.copy()
    label_encoder = LabelEncoder()
    df_prepared['label_encoded'] = label_encoder.fit_transform(df_prepared['label'])

    scaler = StandardScaler()
    df_prepared[numerical_features] = scaler.fit_transform(df_prepared[numerical_features])
    return df_prepared, label_encoder, scaler

def split_data(df_prepared):
    X = df_prepared.drop(['label', 'label_encoded'], axis=1)
    y = df_prepared['label_encoded']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_random_forest(X_train, y_train, model_filename):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'criterion': ['gini', 'entropy']
    }
    rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    random_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    random_search.fit(X_train, y_train)
    best_rf_model = RandomForestClassifier(**random_search.best_params_, random_state=42, n_jobs=-1)
    best_rf_model.fit(X_train, y_train)

    joblib.dump(best_rf_model, model_filename)
    print(f"Trained model saved as {model_filename}")
    return best_rf_model

def evaluate_model(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")

def get_user_input(numerical_features):
    feature_ranges = {
        'N': (0, 200), 'P': (0, 100), 'K': (0, 200),
        'temperature': (0, 50), 'humidity': (0, 100),
        'ph': (4, 9), 'rainfall': (0, 500)
    }
    user_input = {}
    for feature in numerical_features:
        min_val, max_val = feature_ranges[feature]
        while True:
            try:
                val = float(input(f"Enter {feature} ({min_val} - {max_val}): "))
                if min_val <= val <= max_val:
                    user_input[feature] = val
                    break
                else:
                    print(f"Value for {feature} must be between {min_val} and {max_val}.")
            except ValueError:
                print("Please enter a valid number.")
    return pd.DataFrame([user_input])

def predict_probabilities(model, scaler, user_data, label_encoder):
    user_data_scaled = scaler.transform(user_data)
    probabilities = model.predict_proba(user_data_scaled)[0] * 100
    class_names = label_encoder.classes_
    crop_probs = dict(zip(class_names, probabilities))
    sorted_probs = sorted(crop_probs.items(), key=lambda x: x[1], reverse=True)

    print("\n--- Crop Recommendation ---")
    for crop, prob in sorted_probs:
        print(f"{crop}: {prob:.2f}%")

    return sorted_probs

def main():
    csv_file = os.path.join(BASE_DIR, 'crop_recommendation_cleaned.csv')
    model_file = os.path.join(BASE_DIR, 'crop_recommendation_model.pkl')
    scaler_file = os.path.join(BASE_DIR, 'crop_recommendation_scaler.pkl')
    label_file = os.path.join(BASE_DIR, 'crop_recommendation_label_encoder.pkl')

    numerical_features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

    try:
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        label_encoder = joblib.load(label_file)
        print("Loaded existing model.")
    except FileNotFoundError:
        print("Model files not found. Training a new model...")
        df, numerical_features = load_and_preprocess_data(csv_file)
        if df is None:
            print("CSV file not found. Exiting.")
            return

        df_prepared, label_encoder, scaler = prepare_data_for_model(df, numerical_features)
        X_train, X_test, y_train, y_test = split_data(df_prepared)
        model = train_random_forest(X_train, y_train, model_file)
        evaluate_model(model, X_test, y_test, label_encoder)

        joblib.dump(scaler, scaler_file)
        joblib.dump(label_encoder, label_file)
        print("Saved scaler and label encoder.")

    # Get input and predict
    user_data = get_user_input(numerical_features)
    sorted_probs = predict_probabilities(model, scaler, user_data, label_encoder)

    # Show top 5 crops in a pie chart
    top_5 = dict(sorted_probs[:5])
    plt.figure(figsize=(8, 8))
    plt.pie(top_5.values(), labels=top_5.keys(), autopct='%1.1f%%', startangle=140)
    plt.title('Top 5 Recommended Crops')
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()