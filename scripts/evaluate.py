import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score

def evaluate_model():
    # Paths for input data and model
    data_path = os.path.join("features", "titanic_features.csv")
    model_path = os.path.join("models", "model.pkl")
    results_path = os.path.join("results", "metrics.txt")
    predictions_path = os.path.join("results", "predictions.csv")

    if not os.path.exists(data_path) or not os.path.exists(model_path):
        print("Required files missing. Run 'make feature' and 'make train' first.")
        return

    # Load data and model
    df = pd.read_csv(data_path)
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Prepare features and target
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y_true = df['Survived']

    # Generate Predictions
    y_pred = model.predict(X)

    # Calculate Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    # Save Predictions to CSV
    df['Predictions'] = y_pred
    df.to_csv(predictions_path, index=False)

    # Save Metrics to TXT
    with open(results_path, 'w') as f:
        f.write(f"Model Evaluation Metrics:\n")
        f.write(f"-------------------------\n")
        f.write(f"Accuracy:  {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall:    {recall:.4f}\n")

    print(f"Evaluation complete. Metrics saved to {results_path}")
    print(f"Predictions saved to {predictions_path}")

if __name__ == "__main__":
    evaluate_model()