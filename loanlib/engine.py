from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    brier_score_loss,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .config import DATASETS, FIELD_LABELS, RECOMMENDATION_HINTS


def infer_numeric_step(series: pd.Series) -> str:
    values = series.dropna()
    if values.empty:
        return "1"
    if pd.api.types.is_integer_dtype(values):
        return "1"
    sample = values.astype(float).head(25)
    if any(abs(value - round(value)) > 1e-9 for value in sample):
        return "0.01"
    return "1"


def is_categorical_dtype(series: pd.Series) -> bool:
    return (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


class LoanApprovalEngine:
    def __init__(self, base_dir: Path | str = Path.cwd(), models_dir: Path | str | None = None) -> None:
        self.base_dir = Path(base_dir)
        self.models_dir = Path(models_dir) if models_dir else (self.base_dir / "models")
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.models: dict[str, Pipeline] = {}
        self.feature_specs: dict[str, list[dict[str, Any]]] = {}
        self.dataset_profiles: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.dataset_info: dict[str, dict[str, Any]] = {}

    def _get_dataset_path(self, loan_key: str) -> Path:
        meta = DATASETS[loan_key]
        return self.base_dir / meta["file"]

    def train_loan(self, loan_key: str) -> dict[str, Any]:
        dataset_path = self._get_dataset_path(loan_key)
        frame = pd.read_csv(dataset_path)
        if "approved" not in frame.columns:
            raise ValueError("Dataset must contain an 'approved' column for supervised training.")

        X = frame.drop(columns=["approved"])
        y = frame["approved"].astype(int)

        cat_cols = [column for column in X.columns if is_categorical_dtype(X[column])]
        num_cols = [column for column in X.columns if column not in cat_cols]

        # Preprocessing pipelines
        numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, num_cols),
                ("categorical", categorical_pipeline, cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # Candidate classifiers
        rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
        hgb = HistGradientBoostingClassifier(random_state=42, max_iter=200)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")

        rf_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])
        hgb_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", hgb)])
        lr_pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", lr)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Cross-validate candidate models on training set (ROC AUC)
        try:
            rf_cv = float(cross_val_score(rf_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        except Exception:
            rf_cv = 0.0
        try:
            hgb_cv = float(cross_val_score(hgb_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        except Exception:
            hgb_cv = 0.0
        try:
            lr_cv = float(cross_val_score(lr_pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean())
        except Exception:
            lr_cv = 0.0

        # Fit individual pipelines for quick diagnostics
        rf_pipe.fit(X_train, y_train)
        hgb_pipe.fit(X_train, y_train)
        lr_pipe.fit(X_train, y_train)

        # Evaluate each on holdout
        def eval_pipe(pipe):
            probs = pipe.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            return {
                "accuracy": float(accuracy_score(y_test, preds)),
                "f1": float(f1_score(y_test, preds)),
                "roc_auc": float(roc_auc_score(y_test, probs)),
                "precision": float(precision_score(y_test, preds, zero_division=0)),
                "recall": float(recall_score(y_test, preds, zero_division=0)),
                "brier": float(brier_score_loss(y_test, probs)),
            }

        rf_eval = eval_pipe(rf_pipe)
        hgb_eval = eval_pipe(hgb_pipe)
        lr_eval = eval_pipe(lr_pipe)

        # Soft voting ensemble of candidates
        voters = [("rf", rf_pipe), ("hgb", hgb_pipe), ("lr", lr_pipe)]
        ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=-1)

        # Calibrate probabilities for the ensemble (more trustworthy probabilities)
        calibrated = CalibratedClassifierCV(estimator=ensemble, cv=cv, method="sigmoid")
        calibrated.fit(X_train, y_train)

        probs = calibrated.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        self.metrics[loan_key] = {
            "accuracy": round(float(accuracy_score(y_test, preds)), 3),
            "f1_score": round(float(f1_score(y_test, preds)), 3),
            "roc_auc": round(float(roc_auc_score(y_test, probs)), 3),
            "precision": round(float(precision_score(y_test, preds, zero_division=0)), 3),
            "recall": round(float(recall_score(y_test, preds, zero_division=0)), 3),
            "brier": round(float(brier_score_loss(y_test, probs)), 4),
            "cv_scores": {
                "rf": round(rf_cv, 3),
                "hgb": round(hgb_cv, 3),
                "lr": round(lr_cv, 3),
            },
        }

        # Refit calibrated ensemble on full dataset
        calibrated_final = CalibratedClassifierCV(estimator=ensemble, cv=cv, method="sigmoid")
        calibrated_final.fit(X, y)

        model_file = self.models_dir / f"{loan_key}_model.joblib"
        joblib.dump(calibrated_final, model_file)

        feature_specs = self._build_feature_specs(X)
        dataset_profile = self._build_dataset_profile(frame)
        dataset_info = {
            "label": DATASETS[loan_key]["label"],
            "rows": int(len(frame)),
            "approval_rate": round(float(y.mean() * 100), 1),
        }

        meta = {
            "feature_specs": feature_specs,
            "dataset_profile": dataset_profile,
            "dataset_info": dataset_info,
            "metrics": self.metrics[loan_key],
        }
        meta_file = self.models_dir / f"{loan_key}_meta.joblib"
        joblib.dump(meta, meta_file)

        # store in memory
        self.models[loan_key] = calibrated_final
        self.feature_specs[loan_key] = feature_specs
        self.dataset_profiles[loan_key] = dataset_profile
        self.dataset_info[loan_key] = dataset_info

        return meta

    def load_model(self, loan_key: str, train_if_missing: bool = False) -> Pipeline:
        """Load a trained model for `loan_key` from disk.

        If `train_if_missing` is True and a serialized model is not present,
        attempt to train the model from the dataset and save it.
        """
        model_file = self.models_dir / f"{loan_key}_model.joblib"
        meta_file = self.models_dir / f"{loan_key}_meta.joblib"
        if not model_file.exists():
            if train_if_missing:
                dataset_path = self._get_dataset_path(loan_key)
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset file not found for {loan_key}: {dataset_path}")
                # train and persist the model
                self.train_loan(loan_key)
            else:
                raise FileNotFoundError(f"Model file not found for {loan_key}: {model_file}")

        model = joblib.load(model_file)
        self.models[loan_key] = model

        if meta_file.exists():
            meta = joblib.load(meta_file)
            self.feature_specs[loan_key] = meta.get("feature_specs", [])
            self.dataset_profiles[loan_key] = meta.get("dataset_profile", {})
            self.dataset_info[loan_key] = meta.get("dataset_info", {})
            self.metrics[loan_key] = meta.get("metrics", {})
        else:
            # compute metadata lazily
            self.prepare_metadata(loan_key)

        return model

    def prepare_metadata(self, loan_key: str) -> None:
        dataset_path = self._get_dataset_path(loan_key)
        frame = pd.read_csv(dataset_path)
        X = frame.drop(columns=["approved"]) if "approved" in frame.columns else frame.copy()
        self.feature_specs[loan_key] = self._build_feature_specs(X)
        if "approved" in frame.columns:
            self.dataset_profiles[loan_key] = self._build_dataset_profile(frame)
            self.dataset_info[loan_key] = {
                "label": DATASETS[loan_key]["label"],
                "rows": int(len(frame)),
                "approval_rate": round(float(frame["approved"].mean() * 100), 1),
            }

    def parse_form_values(self, loan_key: str, raw_values: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        if loan_key not in self.feature_specs:
            self.prepare_metadata(loan_key)

        # First pass: parse all available inputs based on model feature specs
        for spec in self.feature_specs[loan_key]:
            name = spec["name"]
            raw_value = raw_values.get(name, "")
            if spec["type"] == "categorical":
                parsed[name] = str(raw_value)
            else:
                try:
                    parsed[name] = float(raw_value) if raw_value != "" else None
                except (ValueError, TypeError):
                    parsed[name] = None

        # Helper to retrieve component values from form even if not in model features
        def get_val(keys):
            for k in keys:
                # Check already parsed values first
                if parsed.get(k) is not None:
                    return parsed[k]
                # Then check raw_values (inputs not used as direct features)
                rv = raw_values.get(k, "")
                if rv != "":
                    try: return float(rv)
                    except (ValueError, TypeError): pass
            return None

        # Auto-calculate Ratios if they are missing in input but components exist
        # 1. LTV Ratio (Loan-to-Value)
        if parsed.get("ltv_ratio") is None and "ltv_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            loan = get_val(["loan_amount", "loan_amount_requested", "requested_amount"])
            asset = get_val(["property_value", "car_price", "bike_price", "collateral_value"])
            if loan is not None and asset and asset > 0:
                parsed["ltv_ratio"] = round(loan / asset, 4)

        # 2. DTI Ratio (Debt-to-Income)
        if parsed.get("dti_ratio") is None and "dti_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            emi = get_val(["existing_monthly_emi"]) or 0.0
            income = get_val(["annual_income", "income"])
            if income and income > 0:
                parsed["dti_ratio"] = round(emi / (income / 12), 4)

        # 3. FOIR Ratio (Fixed Obligation to Income Ratio)
        if parsed.get("foir_ratio") is None and "foir_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            emi = get_val(["existing_monthly_emi"]) or 0.0
            income = get_val(["total_annual_income"])
            if income and income > 0:
                parsed["foir_ratio"] = round(emi / (income / 12), 4)

        # 4. DSCR (Debt Service Coverage Ratio)
        if parsed.get("dscr") is None and "dscr" in [s["name"] for s in self.feature_specs[loan_key]]:
            profit = get_val(["net_profit"])
            emi = get_val(["existing_monthly_emi"])
            if profit is not None and emi and emi > 0:
                parsed["dscr"] = round(profit / (emi * 12), 4)

        # Final pass: fill remaining missing numeric fields with median
        for spec in self.feature_specs[loan_key]:
            name = spec["name"]
            if spec["type"] == "numeric" and parsed.get(name) is None:
                parsed[name] = float(spec.get("median", 0.0))

        return parsed

    def predict(self, loan_key: str, values: dict[str, Any]) -> dict[str, Any]:
        if loan_key not in self.models:
            # attempt to load existing model or train on-demand
            self.load_model(loan_key, train_if_missing=True)

        frame = pd.DataFrame([values])
        model = self.models[loan_key]
        probability = float(model.predict_proba(frame)[0][1])
        prediction = int(probability >= 0.5)

        if loan_key not in self.feature_specs:
            self.prepare_metadata(loan_key)

        insights = self._generate_insights(loan_key, values, probability, prediction)
        return {
            "prediction": prediction,
            "probability": round(probability * 100, 2),
            "status": "Approved" if prediction == 1 else "Rejected",
            "confidence_band": self._confidence_band(probability),
            "insights": insights,
        }

    def predict_batch(self, loan_key: str, frame: pd.DataFrame) -> pd.DataFrame:
        expected_columns = [spec["name"] for spec in self.feature_specs[loan_key]]
        missing_columns = [column for column in expected_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

        ordered_frame = frame[expected_columns].copy()
        model = self.models[loan_key]
        probabilities = model.predict_proba(ordered_frame)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        enriched = frame.copy()
        enriched["approval_probability"] = (probabilities * 100).round(2)
        enriched["predicted_status"] = ["Approved" if value == 1 else "Rejected" for value in predictions]
        return enriched

    def _confidence_band(self, probability: float) -> str:
        if probability >= 0.8 or probability <= 0.2:
            return "High confidence"
        if probability >= 0.65 or probability <= 0.35:
            return "Medium confidence"
        return "Borderline case"

    def _build_feature_specs(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for column in frame.columns:
            if is_categorical_dtype(frame[column]):
                options = sorted(str(value) for value in frame[column].dropna().unique())
                specs.append(
                    {
                        "name": column,
                        "label": FIELD_LABELS.get(column, column.replace("_", " ").title()),
                        "type": "categorical",
                        "options": options,
                    }
                )
                continue

            numeric_series = pd.to_numeric(frame[column], errors="coerce")
            specs.append(
                {
                    "name": column,
                    "label": FIELD_LABELS.get(column, column.replace("_", " ").title()),
                    "type": "numeric",
                    "min": round(float(numeric_series.min()), 2) if not numeric_series.dropna().empty else 0.0,
                    "max": round(float(numeric_series.max()), 2) if not numeric_series.dropna().empty else 0.0,
                    "median": round(float(numeric_series.median()), 2) if not numeric_series.dropna().empty else 0.0,
                    "step": infer_numeric_step(numeric_series),
                }
            )
        return specs

    def _build_dataset_profile(self, frame: pd.DataFrame) -> dict[str, Any]:
        approved = frame[frame["approved"] == 1]
        rejected = frame[frame["approved"] == 0]
        profile: dict[str, Any] = {
            "overall_approval_rate": float(frame["approved"].mean()),
            "numeric": {},
            "categorical": {},
        }

        for column in frame.columns:
            if column == "approved":
                continue
            if is_categorical_dtype(frame[column]):
                category_rates = frame.groupby(column, dropna=False)["approved"].mean().to_dict()
                profile["categorical"][column] = {str(key): float(value) for key, value in category_rates.items()}
            else:
                numeric_series = pd.to_numeric(frame[column], errors="coerce")
                spread = float(numeric_series.max() - numeric_series.min()) or 1.0
                profile["numeric"][column] = {
                    "approved_median": float(pd.to_numeric(approved[column], errors="coerce").median()),
                    "rejected_median": float(pd.to_numeric(rejected[column], errors="coerce").median()),
                    "min": float(numeric_series.min()),
                    "max": float(numeric_series.max()),
                    "spread": spread,
                }
        return profile

    def _generate_insights(
        self,
        loan_key: str,
        values: dict[str, Any],
        probability: float,
        prediction: int,
    ) -> dict[str, Any]:
        profile = self.dataset_profiles.get(loan_key, {})
        factors: list[dict[str, Any]] = []

        for spec in self.feature_specs[loan_key]:
            name = spec["name"]
            label = spec["label"]
            value = values[name]
            if spec["type"] == "numeric":
                stats = profile["numeric"][name]
                value_num = float(value)
                approved_distance = abs(value_num - stats["approved_median"]) / stats["spread"]
                rejected_distance = abs(value_num - stats["rejected_median"]) / stats["spread"]
                score = rejected_distance - approved_distance
                factors.append(
                    {
                        "feature": name,
                        "label": label,
                        "value": value_num,
                        "score": score,
                        "detail": (
                            f"{label} is closer to the approved profile median "
                            f"({stats['approved_median']:.2f}) than the rejected median "
                            f"({stats['rejected_median']:.2f})."
                            if score >= 0
                            else f"{label} is closer to the rejected profile median "
                            f"({stats['rejected_median']:.2f}) than the approved median "
                            f"({stats['approved_median']:.2f})."
                        ),
                    }
                )
            else:
                category_rates = profile["categorical"].get(name, {})
                option_rate = category_rates.get(str(value), profile.get("overall_approval_rate", 0.0))
                score = option_rate - profile.get("overall_approval_rate", 0.0)
                factors.append(
                    {
                        "feature": name,
                        "label": label,
                        "value": str(value),
                        "score": score,
                        "detail": (
                            f"{label}='{value}' historically shows a stronger approval rate "
                            f"({option_rate * 100:.1f}%) than the overall dataset."
                            if score >= 0
                            else f"{label}='{value}' historically shows a weaker approval rate "
                            f"({option_rate * 100:.1f}%) than the overall dataset."
                        ),
                    }
                )

        strengths = [factor for factor in sorted(factors, key=lambda item: item["score"], reverse=True) if factor["score"] > 0][:3]
        concerns = [factor for factor in sorted(factors, key=lambda item: item["score"]) if factor["score"] < 0][:3]

        summary = self._build_summary(loan_key, prediction, probability, strengths, concerns)
        recommendations = self._recommend(concerns)
        return {
            "summary": summary,
            "strengths": strengths,
            "concerns": concerns,
            "recommendations": recommendations,
        }

    def _build_summary(
        self,
        loan_key: str,
        prediction: int,
        probability: float,
        strengths: list[dict[str, Any]],
        concerns: list[dict[str, Any]],
    ) -> str:
        loan_name = DATASETS[loan_key]["label"]
        verdict = "approved" if prediction == 1 else "rejected"
        intro = (
            f"The model estimates this {loan_name.lower()} application is likely to be {verdict} "
            f"with an approval probability of {probability * 100:.1f}%.")

        support_text = (
            "The strongest supporting signals are "
            + ", ".join(factor["label"].lower() for factor in strengths[:2])
            + "."
            if strengths
            else "There are no standout positive signals beyond the baseline pattern in the dataset."
        )

        concern_text = (
            "The biggest risks are "
            + ", ".join(factor["label"].lower() for factor in concerns[:2])
            + "."
            if concerns
            else "No major risk factor stands out compared with historical rejected cases."
        )

        return " ".join([intro, support_text, concern_text])

    def _recommend(self, concerns: list[dict[str, Any]]) -> list[str]:
        suggestions: list[str] = []
        for concern in concerns:
            feature = concern["feature"]
            suggestion = RECOMMENDATION_HINTS.get(feature)
            if suggestion and suggestion not in suggestions:
                suggestions.append(suggestion)
        if not suggestions:
            suggestions.append("Maintain the current profile and provide complete supporting documents.")
        return suggestions[:3]
