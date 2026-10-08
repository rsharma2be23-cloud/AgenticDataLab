import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

class ModelTools:
    def __init__(self, output_dir="models"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    # ----------------------------------------------------
    # AUTOMATICALLY detect whether regression or classification
    # ----------------------------------------------------
    def _detect_task(self, y):
        if pd.api.types.is_numeric_dtype(y):
            if y.nunique() > 20:
                return "regression"
            else:
                return "classification"
        return "classification"

    # ----------------------------------------------------
    # PREPROCESSOR: encode categorical + scale numeric cols
    # ----------------------------------------------------
    def _build_preprocessor(self, X):
        numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, cat_cols),
        ])

        return preprocessor

    # ----------------------------------------------------
    # SCORING
    # ----------------------------------------------------
    def _score(self, task, y_true, y_pred):
        if task == "classification":
            return {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "f1_score": float(f1_score(y_true, y_pred, average="weighted")),
            }
        else:
            return {
                "mse": float(mean_squared_error(y_true, y_pred)),
                "r2": float(r2_score(y_true, y_pred)),
            }

    # ----------------------------------------------------
    # MAIN TRAIN FUNCTION
    # ----------------------------------------------------
    def train(self, df, target_col):
        try:
            X = df.drop(columns=[target_col])
            y = df[target_col]

            task = self._detect_task(y)
            preprocessor = self._build_preprocessor(X)

            # -------------------------------
            # SELECT MODELS
            # -------------------------------
            if task == "classification":
                models = {
                    "rf": RandomForestClassifier(),
                    "logreg": LogisticRegression(max_iter=500),
                }
            else:
                models = {
                    "rf": RandomForestRegressor(),
                    "ridge": Ridge(),
                    "svr": SVR(),
                }

            best_model = None
            best_score = -999999
            best_metrics = None

            # -------------------------------
            # TRAIN MULTIPLE MODELS
            # -------------------------------
            for name, model in models.items():
                pipe = Pipeline([
                    ("prep", preprocessor),
                    ("model", model),
                ])

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                pipe.fit(X_train, y_train)
                preds = pipe.predict(X_test)

                metrics = self._score(task, y_test, preds)

                score = metrics.get("r2") if task == "regression" else metrics.get("f1_score")

                if score is None:
                    score = -metrics.get("mse", 999999)

                if score > best_score:
                    best_score = score
                    best_metrics = metrics
                    best_model = pipe
                    best_name = name
                    best_preds = preds

            # -------------------------------
            # SAVE MODEL
            # -------------------------------
            model_path = os.path.join(self.output_dir, f"{best_name}_model.pkl")
            pickle.dump(best_model, open(model_path, "wb"))

            # -------------------------------
            # RETURN RESULTS
            # -------------------------------
            return {
                "status": "success",
                "task_type": task,
                "model_name": best_name,
                "model_path": model_path,
                "metrics": best_metrics,
                "sample_predictions": best_preds[:10].tolist(),
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}
