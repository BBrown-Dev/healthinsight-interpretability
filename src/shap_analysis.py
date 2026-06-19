import shap
import numpy as np
from src.utils import log, Timer, save_plot

def run_shap_analysis(model, X_test, feature_names):
    with Timer("SHAP Analysis"):
        """Run SHAP analysis for both global and local explanations."""

        # Global SHAP Analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Global SHAP Summary
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        sorted_idx = np.argsort(mean_abs)[::-1]

        log("Top SHAP Contributors (Global):")
        for idx in sorted_idx[:5]:
            log(f"    {feature_names[idx]}: {mean_abs[idx]:.4f}")

        # SHAP Summary Plot
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        save_plot("shap_summary.png")

        # SHAP Dependence Plot for Glucose
        shap.dependence_plot(feature_names[3], shap_values, X_test, show=False)
        save_plot("shap_dependence_glucose.png")

        # Local SHAP Explanation
        log("Generating Local SHAP Explanation for Patient #12...")
        patient_index = 12
        patient_data = X_test.iloc[[patient_index]]
        patient_shap = shap_values[patient_index]

        # Sort contributions
        sorted_local_idx = np.argsort(np.abs(patient_shap))[::-1]

        log(f"Local SHAP Explanation for Patient #{patient_index}:")
        for idx in sorted_local_idx[:5]:
            direction = "increases risk" if patient_shap[idx] > 0 else "decreases risk"
            log(f"    {feature_names[idx]}: {patient_shap[idx]:.4f} ({direction})")

        # Save force plot
        try:
            shap.force_plot(
                explainer.expected_value,
                patient_shap,
                patient_data,
                matplotlib=True,
                show=False
            )
            save_plot("shap_local_patient12.png")
        except Exception:
            log("Force plot could not be generated in this environment.")

        log("SHAP analysis complete")