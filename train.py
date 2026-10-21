import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # Step 1: Reading training data
    df = pd.read_csv("data/patients.csv")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    # Step 2: Train Test split 
    X = df.drop("target", axis=1)
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 3: Treating missing values


    # Step 4: Categorical encoding


    # Step 5: Model training and evaluation


    # Step 6: Save the trained model to score it on new population



if __name__ == "__main__":
    main()