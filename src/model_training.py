from sklearn.metrics import accuracy_score, roc_auc_score
from src.utils import log, Timer
import xgboost as xgb

def train_xgboost_model(X_train, y_train):
    with Timer("Model Training"):
        """Train an XGBoost classifier on the training data."""

        # Initialize XGBoost model with specified hyperparameters
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        # Evaluate training performance
        preds = model.predict(X_train)
        acc = accuracy_score(y_train, preds)
        auc = roc_auc_score(y_train, preds)

        # Log training metrics
        log(f"Training Accuracy: {acc:.4f}")
        log(f"Training AUC: {auc:.4f}")

        log("XGBoost model training complete")
        return model