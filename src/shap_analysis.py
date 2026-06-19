import shap
from src.utils import log, Timer, save_plot

def run_shap_analysis(model, X_test, feature_names):
    with Timer("SHAP Analysis"):
        """Run SHAP analysis to explain model predictions."""

        # Create SHAP explainer and compute SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Summary plot of SHAP values
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        save_plot("shap_summary.png")

        # Dependence plot for a specific feature (e.g., glucose level)
        shap.dependence_plot(feature_names[3], shap_values, X_test, show=False)
        save_plot("shap_dependence_glucose.png")

        log("SHAP analysis complete")