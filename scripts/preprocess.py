import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def preprocess_data():
    # Paths based on project structure [cite: 13, 14]
    input_path = os.path.join("data", "raw", "titanic.csv")
    output_path = os.path.join("data", "processed", "titanic_cleaned.csv")
    
    if not os.path.exists(input_path):
        print("Raw data not found! Please run 'make download-data' first.")
        return

    df = pd.read_csv(input_path)

    # Manual requirement: Handle missing values [cite: 30]
    df['Age'] = df['Age'].fillna(df['Age'].median())
    
    # Manual requirement: Encode categorical variables (Sex and Embarked) [cite: 30]
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'].astype(str))
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
        df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))

    # Ensure output directory exists [cite: 31]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the processed dataset
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()