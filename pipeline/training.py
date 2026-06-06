import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


class ModelTrainer:
    AVAILABLE_MODELS = {
        'logreg': LogisticRegression,
        'rf': RandomForestClassifier,
        'xgb': xgb.XGBClassifier,
        'lgbm': lgb.LGBMClassifier,
    }

    DEFAULT_PARAMS = {
        'logreg': {'max_iter': 1000, 'random_state': 42},
        'rf': {'n_estimators': 200, 'random_state': 42, 'n_jobs': -1},
        'xgb': {
            'n_estimators': 200, 'random_state': 42,
            'eval_metric': 'mlogloss', 'n_jobs': -1,
        },
        'lgbm': {
            'n_estimators': 200, 'random_state': 42,
            'n_jobs': -1, 'verbose': -1,
        },
    }

    def __init__(self, model_name, params=None):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {list(self.AVAILABLE_MODELS)}"
            )
        self.model_name = model_name
        self.params = {**self.DEFAULT_PARAMS[model_name], **(params or {})}
        self.model = self.AVAILABLE_MODELS[model_name](**self.params)
        self.is_trained = False

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model belum di-train. Call .train() dulu.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model belum di-train. Call .train() dulu.")
        return self.model.predict_proba(X)

    def get_params(self):
        return self.params

    def save(self, path, compress=3):
        joblib.dump(self.model, path, compress=compress)

    @classmethod
    def load(cls, path):
        return joblib.load(path)
