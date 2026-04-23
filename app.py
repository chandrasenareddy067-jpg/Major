from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import os

import pandas as pd
import joblib
from flask import Flask, flash, g, jsonify, redirect, render_template, request, send_file, url_for, session
import json
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent

DATASETS = {
    "agriculture": {"label": "Agriculture Loan", "file": "data_sets/agriculture_loan_data.csv"},
    "bike": {"label": "Bike Loan", "file": "data_sets/bike_loans.csv"},
    "business": {"label": "Business Loan", "file": "data_sets/business_loans.csv"},
    "car": {"label": "Car Loan", "file": "data_sets/car_loans.csv"},
    "education": {"label": "Education Loan", "file": "data_sets/education_loans.csv"},
    "home": {"label": "Home Loan", "file": "data_sets/home_loans.csv"},
    "personal": {"label": "Personal Loan", "file": "data_sets/personal_loans.csv"},
}

FIELD_LABELS = {
    "age": "Applicant Age",
    "annual_income": "Annual Income",
    "annual_revenue": "Annual Revenue",
    "annual_farm_revenue": "Annual Farm Revenue",
    "alt_annual_income": "Alternate Annual Income",
    "alt_income_source": "Alternate Income Source",
    "applicant_age": "Applicant Age",
    "bike_price": "Bike Price",
    "car_price": "Car Price",
    "car_type": "Car Type",
    "collateral_type": "Collateral Type",
    "collateral_value": "Collateral Value",
    "course_category": "Course Category",
    "credit_score": "Credit Score",
    "down_payment": "Down Payment",
    "dscr": "DSCR",
    "dti_ratio": "DTI Ratio",
    "emp_status": "Employment Status",
    "employment_tenure_years": "Employment Tenure (Years)",
    "existing_monthly_emi": "Existing Monthly EMI",
    "farmer_age": "Farmer Age",
    "foir_ratio": "FOIR Ratio",
    "gpa": "GPA",
    "income": "Annual Income",
    "industry_type": "Industry Type",
    "land_acres": "Land Size (Acres)",
    "loan_amount": "Loan Amount",
    "loan_amount_requested": "Loan Amount Requested",
    "loan_purpose": "Loan Purpose",
    "loan_term": "Loan Term (Months)",
    "loan_term_months": "Loan Term (Months)",
    "ltv_ratio": "LTV Ratio",
    "net_profit": "Net Profit",
    "owner_credit_score": "Owner Credit Score",
    "parent_annual_income": "Parent Annual Income",
    "parent_credit_score": "Parent Credit Score",
    "property_value": "Property Value",
    "requested_amount": "Requested Amount",
    "tenure_years": "Tenure (Years)",
    "total_annual_income": "Total Annual Income",
    "uni_ranking": "University Ranking",
    "years_experience": "Years of Experience",
    "years_in_business": "Years in Business",
}

RECOMMENDATION_HINTS = {
    "credit_score": "improve the credit profile or add a stronger co-applicant",
    "owner_credit_score": "improve the credit profile before reapplying",
    "parent_credit_score": "strengthen the guarantor or parent credit profile",
    "annual_income": "show higher stable income or lower the requested amount",
    "income": "show higher stable income or reduce the financed amount",
    "annual_revenue": "provide stronger revenue proof or request a smaller loan",
    "annual_farm_revenue": "provide stronger farm revenue evidence or request a smaller amount",
    "net_profit": "improve profitability and supporting business documents",
    "down_payment": "increase the down payment to lower financing risk",
    "ltv_ratio": "lower the LTV by increasing upfront contribution",
    "dti_ratio": "reduce existing debt obligations before reapplying",
    "foir_ratio": "lower current obligations to improve repayment capacity",
    "existing_monthly_emi": "reduce ongoing EMI burden before taking a new loan",
    "loan_amount": "request a smaller loan amount",
    "loan_amount_requested": "request a smaller loan amount",
    "requested_amount": "request a smaller business loan amount",
    "property_value": "choose a lower-risk property or increase the down payment",
    "collateral_value": "offer stronger collateral coverage",
    "dscr": "improve cash flow coverage for the requested installment",
    "gpa": "strengthen academic performance or add a stronger repayment plan",
    "uni_ranking": "target a better-ranked program or show stronger funding support",
}


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
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.models_dir = base_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        self.models: dict[str, Pipeline] = {}
        self.feature_specs: dict[str, list[dict[str, Any]]] = {}
        self.dataset_profiles: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self.dataset_info: dict[str, dict[str, Any]] = {}
        self._load_everything()

    def _load_everything(self) -> None:
        for loan_key, meta in DATASETS.items():
            model_path = self.models_dir / f"{loan_key}_model.joblib"
            meta_path = self.models_dir / f"{loan_key}_meta.joblib"

            if model_path.exists() and meta_path.exists():
                # Load pre-trained model and metadata
                self.models[loan_key] = joblib.load(model_path)
                metadata = joblib.load(meta_path)
                self.feature_specs[loan_key] = metadata["feature_specs"]
                self.dataset_profiles[loan_key] = metadata["dataset_profile"]
                self.dataset_info[loan_key] = metadata["dataset_info"]
                self.metrics[loan_key] = metadata["metrics"]
                continue

            # Fallback: Train if files don't exist
            frame = pd.read_csv(self.base_dir / meta["file"])
            X = frame.drop(columns=["approved"])
            y = frame["approved"].astype(int)

            cat_cols = [column for column in X.columns if is_categorical_dtype(X[column])]
            num_cols = [column for column in X.columns if column not in cat_cols]

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numeric",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scaler", StandardScaler()),
                            ]
                        ),
                        num_cols,
                    ),
                    (
                        "categorical",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("encoder", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        cat_cols,
                    ),
                ]
            )

            model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y,
            )

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1]
            self.metrics[loan_key] = {
                "accuracy": round(float(accuracy_score(y_test, predictions)), 3),
                "f1_score": round(float(f1_score(y_test, predictions)), 3),
                "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 3),
            }

            # Refit on the full dataset for the best inference-time model.
            model.fit(X, y)
            self.models[loan_key] = model
            self.feature_specs[loan_key] = self._build_feature_specs(X)
            self.dataset_profiles[loan_key] = self._build_dataset_profile(frame)
            self.dataset_info[loan_key] = {
                "label": meta["label"],
                "rows": int(len(frame)),
                "approval_rate": round(float(y.mean() * 100), 1),
            }

            # Save for future use
            joblib.dump(self.models[loan_key], model_path)
            joblib.dump({
                "feature_specs": self.feature_specs[loan_key],
                "dataset_profile": self.dataset_profiles[loan_key],
                "dataset_info": self.dataset_info[loan_key],
                "metrics": self.metrics[loan_key]
            }, meta_path)

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
                    "min": round(float(numeric_series.min()), 2),
                    "max": round(float(numeric_series.max()), 2),
                    "median": round(float(numeric_series.median()), 2),
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
                category_rates = (
                    frame.groupby(column, dropna=False)["approved"].mean().to_dict()
                )
                profile["categorical"][column] = {
                    str(key): float(value) for key, value in category_rates.items()
                }
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

    def parse_form_values(self, loan_key: str, raw_values: dict[str, Any]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
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

        def get_val(keys):
            for k in keys:
                if parsed.get(k) is not None:
                    return parsed[k]
                rv = raw_values.get(k, "")
                if rv != "":
                    try: return float(rv)
                    except (ValueError, TypeError): pass
            return None

        # Auto-calculate Ratios
        if parsed.get("ltv_ratio") is None and "ltv_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            loan = get_val(["loan_amount", "loan_amount_requested", "requested_amount"])
            asset = get_val(["property_value", "car_price", "bike_price", "collateral_value"])
            if loan is not None and asset and asset > 0:
                parsed["ltv_ratio"] = round(loan / asset, 4)

        if parsed.get("dti_ratio") is None and "dti_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            emi = get_val(["existing_monthly_emi"]) or 0.0
            income = get_val(["annual_income", "income"])
            if income and income > 0:
                parsed["dti_ratio"] = round(emi / (income / 12), 4)

        if parsed.get("foir_ratio") is None and "foir_ratio" in [s["name"] for s in self.feature_specs[loan_key]]:
            emi = get_val(["existing_monthly_emi"]) or 0.0
            income = get_val(["total_annual_income"])
            if income and income > 0:
                parsed["foir_ratio"] = round(emi / (income / 12), 4)

        if parsed.get("dscr") is None and "dscr" in [s["name"] for s in self.feature_specs[loan_key]]:
            profit = get_val(["net_profit"])
            emi = get_val(["existing_monthly_emi"])
            if profit is not None and emi and emi > 0:
                parsed["dscr"] = round(profit / (emi * 12), 4)

        for spec in self.feature_specs[loan_key]:
            name = spec["name"]
            if spec["type"] == "numeric" and parsed.get(name) is None:
                parsed[name] = float(spec.get("median", 0.0))

        return parsed

    def predict(self, loan_key: str, values: dict[str, Any]) -> dict[str, Any]:
        frame = pd.DataFrame([values])
        model = self.models[loan_key]
        probability = float(model.predict_proba(frame)[0][1])
        prediction = int(probability >= 0.5)
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

    def _generate_insights(
        self,
        loan_key: str,
        values: dict[str, Any],
        probability: float,
        prediction: int,
    ) -> dict[str, Any]:
        profile = self.dataset_profiles[loan_key]
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
                category_rates = profile["categorical"][name]
                option_rate = category_rates.get(str(value), profile["overall_approval_rate"])
                score = option_rate - profile["overall_approval_rate"]
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
            f"with an approval probability of {probability * 100:.1f}%."
        )

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


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key-for-local-use-only")
engine = LoanApprovalEngine(BASE_DIR)
from auth import auth_bp
app.register_blueprint(auth_bp)
import db as db_module

# Ensure database tables are created if missing
try:
    db_module.init_db()
except Exception:
    # don't crash server on DB init errors; log if needed
    pass
from admin import admin_bp
app.register_blueprint(admin_bp)


def _load_current_user() -> dict[str, Any] | None:
    cached_user = getattr(g, "_current_user", None)
    if cached_user is not None or hasattr(g, "_current_user_loaded"):
        return cached_user

    user_id = session.get("user_id")
    g._current_user_loaded = True
    if not user_id:
        g._current_user = None
        return None

    from models import User

    db = db_module.SessionLocal()
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            session.pop("user_id", None)
            g._current_user = None
            return None

        g._current_user = {
            "id": user.id,
            "email": user.email,
            "phone": user.phone,
            "created_at": user.created_at,
            "is_active": bool(user.is_active),
        }
        return g._current_user
    finally:
        db.close()


def _require_current_user():
    current_user = _load_current_user()
    if current_user:
        return current_user

    flash("Please log in to view your account.", "error")
    return redirect(url_for("auth.login"))


@app.context_processor
def inject_current_user():
    return {"current_user": _load_current_user()}


@app.get("/")
def index() -> str:
    selected_loan = request.args.get("loan_type", next(iter(DATASETS)))
    return render_page(selected_loan=selected_loan, current_values={}, prediction_result=None)


@app.post("/predict")
def predict_form():
    selected_loan = request.form.get("loan_type", next(iter(DATASETS)))
    current_values = {key: value for key, value in request.form.items() if key != "loan_type"}
    try:
        parsed_values = engine.parse_form_values(selected_loan, current_values)
        prediction_result = engine.predict(selected_loan, parsed_values)
        # persist application + insights
        try:
            from models import Application, ApplicationInsight
            db = db_module.SessionLocal()
            app_record = Application(
                user_id=session.get("user_id"),
                loan_type=selected_loan,
                form_data=json.dumps(parsed_values),
                predicted_status=prediction_result.get("status"),
                predicted_probability=prediction_result.get("probability"),
                ltv_ratio=parsed_values.get("ltv_ratio"),
                dti_ratio=parsed_values.get("dti_ratio"),
                foir_ratio=parsed_values.get("foir_ratio"),
                dscr=parsed_values.get("dscr"),
            )
            db.add(app_record)
            db.commit()
            db.refresh(app_record)

            insights = prediction_result.get("insights", {})
            strengths = insights.get("strengths", [])
            concerns = insights.get("concerns", [])
            recommendations = insights.get("recommendations", [])

            main_positive = strengths[0]["label"] if strengths else None
            main_negative = concerns[0]["label"] if concerns else None
            main_reco = recommendations[0] if recommendations else None

            insight_record = ApplicationInsight(
                application_id=app_record.id,
                positive_signals=json.dumps([s.get("label") for s in strengths]),
                negative_signals=json.dumps([c.get("label") for c in concerns]),
                recommendations=json.dumps(recommendations),
                main_positive=main_positive,
                main_negative=main_negative,
                main_recommendation=main_reco,
            )
            db.add(insight_record)
            db.commit()
        except Exception:
            # don't break prediction if DB persist fails
            pass
        finally:
            try:
                db.close()
            except Exception:
                pass

        return render_page(
            selected_loan=selected_loan,
            current_values=current_values,
            prediction_result=prediction_result,
        )
    except ValueError as exc:
        flash(f"Please complete the form with valid values. {exc}", "error")
        return render_page(selected_loan=selected_loan, current_values=current_values, prediction_result=None), 400


@app.post("/api/predict")
def predict_api():
    payload = request.get_json(silent=True) or {}
    selected_loan = payload.get("loan_type")
    if selected_loan not in DATASETS:
        return jsonify({"error": "Invalid loan type."}), 400
    try:
        values = engine.parse_form_values(selected_loan, payload.get("values", {}))
        result = engine.predict(selected_loan, values)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/batch-predict")
def batch_predict():
    selected_loan = request.form.get("loan_type", next(iter(DATASETS)))
    uploaded_file = request.files.get("batch_file")
    if uploaded_file is None or uploaded_file.filename == "":
        flash("Upload a CSV file to run batch predictions.", "error")
        return redirect(url_for("index", loan_type=selected_loan))

    try:
        frame = pd.read_csv(uploaded_file)
        enriched = engine.predict_batch(selected_loan, frame)
    except Exception as exc:
        flash(f"Batch prediction failed. {exc}", "error")
        return redirect(url_for("index", loan_type=selected_loan))

    buffer = BytesIO()
    enriched.to_csv(buffer, index=False)
    buffer.seek(0)
    output_name = f"{selected_loan}_loan_predictions.csv"
    return send_file(
        buffer,
        as_attachment=True,
        download_name=output_name,
        mimetype="text/csv",
    )


@app.get("/profile")
def profile():
    current_user = _require_current_user()
    if not isinstance(current_user, dict):
        return current_user

    from models import Application

    db = db_module.SessionLocal()
    try:
        applications = (
            db.query(Application)
            .filter(Application.user_id == current_user["id"])
            .order_by(Application.created_at.desc())
            .all()
        )
    finally:
        db.close()

    total_applications = len(applications)
    approved_applications = sum(
        1 for application in applications if (application.predicted_status or "").lower() == "approved"
    )
    rejected_applications = sum(
        1 for application in applications if (application.predicted_status or "").lower() == "rejected"
    )
    latest_application = applications[0] if applications else None

    return render_template(
        "profile.html",
        active_page="profile",
        user_profile=current_user,
        total_applications=total_applications,
        approved_applications=approved_applications,
        rejected_applications=rejected_applications,
        latest_application=latest_application,
        loan_labels={key: meta["label"] for key, meta in DATASETS.items()},
    )


@app.get("/applications")
def applications_history():
    current_user = _require_current_user()
    if not isinstance(current_user, dict):
        return current_user

    from models import Application

    db = db_module.SessionLocal()
    try:
        applications = (
            db.query(Application)
            .filter(Application.user_id == current_user["id"])
            .order_by(Application.created_at.desc())
            .all()
        )

        history: list[dict[str, Any]] = []
        for application in applications:
            try:
                form_data = json.loads(application.form_data) if application.form_data else {}
            except json.JSONDecodeError:
                form_data = {}

            insight = application.insight
            positive_signals: list[str] = []
            negative_signals: list[str] = []
            recommendations: list[str] = []
            if insight:
                try:
                    positive_signals = json.loads(insight.positive_signals) if insight.positive_signals else []
                except json.JSONDecodeError:
                    positive_signals = []
                try:
                    negative_signals = json.loads(insight.negative_signals) if insight.negative_signals else []
                except json.JSONDecodeError:
                    negative_signals = []
                try:
                    recommendations = json.loads(insight.recommendations) if insight.recommendations else []
                except json.JSONDecodeError:
                    recommendations = []

            history.append(
                {
                    "id": application.id,
                    "loan_type": application.loan_type,
                    "loan_label": DATASETS.get(application.loan_type, {}).get("label", application.loan_type.title()),
                    "predicted_status": application.predicted_status or "Pending",
                    "predicted_probability": application.predicted_probability,
                    "created_at": application.created_at,
                    "form_data": form_data,
                    "positive_signals": positive_signals,
                    "negative_signals": negative_signals,
                    "recommendations": recommendations,
                }
            )
    finally:
        db.close()

    return render_template(
        "applications.html",
        active_page="applications",
        applications=history,
        user_profile=current_user,
    )


def render_page(
    selected_loan: str,
    current_values: dict[str, Any],
    prediction_result: dict[str, Any] | None,
):
    user_logged_in = bool(session.get("user_id"))
    excluded = {"ltv_ratio", "dti_ratio", "foir_ratio", "dscr"}
    filtered_specs = {k: [s for s in v if s["name"] not in excluded] 
                     for k, v in engine.feature_specs.items()}
    return render_template(
        "index.html",
        datasets=DATASETS,
        selected_loan=selected_loan,
        field_specs=filtered_specs,
        current_values=current_values,
        prediction_result=prediction_result,
        metrics=engine.metrics,
        dataset_info=engine.dataset_info,
        user_logged_in=user_logged_in,
        active_page="home",
    )


if __name__ == "__main__":
    app.run(debug=True)
