import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from src.utils import log, Timer, save_plot

def plot_decision_boundary(model):
    with Timer("Decision Boundary Visualization"):
        """Plot the decision boundary of the model."""
        df = pd.read_csv("data/healthcare_simulated.csv")

        # Use only two features for visualization
        X = df[["Glucose", "BMI"]]
        y = df["Outcome"]

        # Train a simple model for decision boundary visualization
        simple_model = RandomForestClassifier().fit(X, y)

        # Create a mesh grid for plotting
        x_min, x_max = X["Glucose"].min() - 1, X["Glucose"].max() + 1
        y_min, y_max = X["BMI"].min() - 1, X["BMI"].max() + 1

        # Create a mesh grid for plotting
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )

        # Predict the class for each point in the mesh grid
        Z = simple_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the decision boundary and the data points
        plt.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        plt.scatter(X["Glucose"], X["BMI"], c=y, cmap="coolwarm", edgecolor="k")
        plt.xlabel("Glucose")
        plt.ylabel("BMI")
        plt.title("Decision Boundary (Glucose vs BMI)")

        save_plot("decision_boundary.png")
        plt.show()

        log("Decision boundary visualization complete")