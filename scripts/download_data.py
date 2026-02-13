import pandas as pd
import os

def download_titanic():
    # URL to a reliable source of the Titanic dataset
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    
    # Path where the manual says raw data must be stored 
    output_path = os.path.join("data", "raw", "titanic.csv")
    
    print(f"Downloading dataset from {url}...")
    df = pd.read_csv(url)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to the specified folder
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully saved to {output_path}")

if __name__ == "__main__":
    download_titanic()