import shap
import numpy as np
from src.utils import log, Timer, save_plot

def run_shap_analysis(model, X_test, feature_names):
    with Timer("SHAP Analysis"):
        """Run SHAP analysis to explain model predictions."""

        # Create SHAP explainer and compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Calculate mean absolute SHAP values for each feature and sort them
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        # Log the top SHAP contributors
        log("Top SHAP Contributors:")
        for idx in sorted_idx[:5]:
            log(f"    {feature_names[idx]}: {mean_abs[idx]:.4f}")

        # Generate SHAP summary plot
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        save_plot("shap_summary.png")

        # Generate SHAP dependence plot for the top feature
        shap.dependence_plot(feature_names[3], shap_values, X_test, show=False)
        save_plot("shap_dependence_glucose.png")

        log("SHAP analysis complete")