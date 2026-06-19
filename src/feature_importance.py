import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from src.utils import log, save_plot, Timer

def plot_feature_importance(model):
    with Timer("Feature Importance Plot"):
        """Plot feature importance."""

        # Get feature importance scores from the model
        importance = model.get_booster().get_score(importance_type='gain')
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        # Log the sorted feature importance scores
        log("Feature Importance (sorted):")
        for feat, score in sorted_imp:
            log(f"    {feat}: {score:.4f}")

        # Plot feature importance using XGBoost's built-in function
        xgb.plot_importance(model, max_num_features=10)
        plt.title("XGBoost Feature Importance")
        save_plot("feature_importance.png")
        plt.show()
        log("Feature importance plot generated")