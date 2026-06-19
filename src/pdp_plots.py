import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from src.utils import log, save_plot, Timer

def generate_pdp(model, X_train, feature_names):
    with Timer("Partial Dependence Plots"):
        """Plot the partial dependence plot."""
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            model, X_train, [feature_names[1], feature_names[3]], ax=ax
        )
        plt.suptitle("Partial Dependence Plots")
        save_plot("pdp_plots.png")
        plt.show()
        log("PDPs generated")