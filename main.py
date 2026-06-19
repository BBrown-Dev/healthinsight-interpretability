from src.data_simulation import generate_healthcare_data
from src.model_training import train_xgboost_model
from src.feature_importance import plot_feature_importance
from src.pdp_plots import generate_pdp
from src.shap_analysis import run_shap_analysis
from src.decision_boundary import plot_decision_boundary
from src.utils import log

def main():
    """Run main pipeline."""
    log("Starting HealthInsight Interpretability Pipeline")

    # Data Simulation
    X_train, X_test, y_train, y_test, feature_names = generate_healthcare_data()

    # Model Training
    model = train_xgboost_model(X_train, y_train)

    # Interpretability Analyses
    plot_feature_importance(model)
    generate_pdp(model, X_train, feature_names)
    run_shap_analysis(model, X_test, feature_names)
    plot_decision_boundary(model)

    log("Pipeline complete")

if __name__ == "__main__":
    main()