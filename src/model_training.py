import xgboost as xgb
from src.utils import log, Timer

def train_xgboost_model(X_train, y_train):
    with Timer("Model Training"):
        """Train an XGBoost classifier on the provided training data."""
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        log("XGBoost model training complete")
        return model