import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Input path: Output from Phase 4
    input_path = os.path.join("features", "titanic_features.csv")
    # Output path: Requirement for Phase 5
    model_dir = "models"
    model_path = os.path.join(model_dir, "model.pkl")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Please run 'make feature' first.")
        return

    # Load the engineered features
    df = pd.read_csv(input_path)

    # Prepare features (X) and target (y)
    # We use our newly created 'FamilySize' along with standard features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']
    X = df[features]
    y = df['Survived']

    # Initialize and train the Random Forest Classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Ensure the 'models' directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Serialize and save the model using pickle
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model successfully trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()