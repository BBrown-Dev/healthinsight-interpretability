import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.utils import log, ensure_dir, Timer

def generate_healthcare_data(save_csv=True):
    with Timer("Data Simulation"):
        """Simulates a healthcare dataset for binary classification. The dataset includes features such as Age, BMI, Blood Pressure, Glucose, Heart Rate, and Cholesterol. The target variable indicates the presence (1) or absence (0) of a health condition."""
        X, y = make_classification(
            n_samples=2000,
            n_features=6,
            n_informative=4,
            n_redundant=0,
            class_sep=1.5,
            random_state=42
        )

        # Create a DataFrame for better handling and potential CSV export
        feature_names = ["Age", "BMI", "BloodPressure", "Glucose", "HeartRate", "Cholesterol"]
        df = pd.DataFrame(X, columns=feature_names)
        df["Outcome"] = y

        # Save the dataset to a CSV file for future use
        if save_csv:
            ensure_dir("data")
            df.to_csv("data/healthcare_simulated.csv", index=False)
            log("Dataset saved to data/healthcare_simulated.csv")

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            df[feature_names], df["Outcome"], test_size=0.2, random_state=42
        )

        log("Data simulation complete")

        return X_train, X_test, y_train, y_test, feature_names