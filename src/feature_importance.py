import matplotlib.pyplot as plt
import xgboost as xgb
from src.utils import log, save_plot, Timer

def plot_feature_importance(model):
    with Timer("Feature Importance Plot"):
        """Plot the feature importance plot."""
        xgb.plot_importance(model, max_num_features=10)
        plt.title("XGBoost Feature Importance")
        save_plot("feature_importance.png")
        plt.show()
        log("Feature importance plot generated")