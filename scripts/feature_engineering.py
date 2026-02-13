import pandas as pd
import os

def create_features():
    input_path = os.path.join("data", "processed", "titanic_cleaned.csv")
    output_path = os.path.join("features", "titanic_features.csv")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Run 'make preprocess' first.")
        return

    df = pd.read_csv(input_path)

    # Dictionary to map possible column names to our variables
    # This handles both Kaggle and standard Titanic dataset formats
    sib_cols = ['Siblings/Spouses Aboard', 'SibSp']
    par_cols = ['Parents/Children Aboard', 'Parch']

    # Find which column exists in your data
    sib_col = next((c for c in sib_cols if c in df.columns), None)
    par_col = next((c for c in par_cols if c in df.columns), None)

    if sib_col and par_col:
        print(f"Using columns: {sib_col} and {par_col}")
        df['FamilySize'] = df[sib_col] + df[par_col] + 1
    else:
        print("Warning: Could not find sibling or parent columns. Check your CSV headers!")
        # Fallback if names are completely different
        print(f"Available columns: {list(df.columns)}")
        return

    # Create IsAlone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    os.makedirs("features", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature engineering successful! Saved to {output_path}")

if __name__ == "__main__":
    create_features()