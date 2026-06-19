"""
HealthInsight Interpretability Toolkit
--------------------------------------

This package provides utilities for:
- Simulating healthcare datasets
- Training predictive ML models
- Visualizing feature importance
- Generating partial dependence plots (PDPs)
- Running SHAP global and local explanations
- Visualizing decision boundaries

Author: Bruce Brown
Version: 1.0.0
"""

from .data_simulation import generate_healthcare_data
from .model_training import train_xgboost_model
from .feature_importance import plot_feature_importance
from .pdp_plots import generate_pdp
from .shap_analysis import run_shap_analysis
from .decision_boundary import plot_decision_boundary

__all__ = [
    "generate_healthcare_data",
    "train_xgboost_model",
    "plot_feature_importance",
    "generate_pdp",
    "run_shap_analysis",
    "plot_decision_boundary"
]