from pathlib import Path
from io import BytesIO

import json
import os
import pandas as pd
from flask import Flask, flash, jsonify, redirect, render_template, request, send_file, url_for, session

from loanlib.config import DATASETS
from loanlib.engine import LoanApprovalEngine

import db as db_module # type: ignore
from auth import auth_bp
from admin import admin_bp



app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "super-secret-dev-key") # Use environment variable for secret key

engine = LoanApprovalEngine(base_dir=Path('.'))

# Try to load existing models; compute metadata if models are missing
for key in DATASETS:
    try:
        engine.load_model(key)
    except Exception:
        try:
            engine.prepare_metadata(key)
        except Exception:
            # ignore; server can still start and users can train models separately
            pass

# initialize the application database (models will be created)
try:
    db_module.init_db()
except Exception:
    # allow server to continue even if DB initialization fails locally
    pass

# register auth blueprint
app.register_blueprint(auth_bp)

# register admin blueprint
app.register_blueprint(admin_bp)


@app.get("/")
def index() -> str:
    selected_loan = request.args.get("loan_type", next(iter(DATASETS)))
    user_logged_in = bool(session.get("user_id"))
    # Filter out ratio fields from the frontend
    excluded = {"ltv_ratio", "dti_ratio", "foir_ratio", "dscr"}
    filtered_specs = {k: [s for s in v if s["name"] not in excluded] 
                     for k, v in engine.feature_specs.items()}
    return render_template(
        "index.html",
        datasets=DATASETS,
        selected_loan=selected_loan,
        field_specs=filtered_specs,
        current_values={},
        prediction_result=None,
        metrics=engine.metrics,
        dataset_info=engine.dataset_info,
        user_logged_in=user_logged_in,
    )


@app.post("/predict")
def predict_form():
    selected_loan = request.form.get("loan_type", next(iter(DATASETS)))
    current_values = {key: value for key, value in request.form.items() if key != "loan_type"}
    try:
        parsed_values = engine.parse_form_values(selected_loan, current_values)
        prediction_result = engine.predict(selected_loan, parsed_values)
        # persist application for logged-in users
        user_id = session.get("user_id")
        if user_id:
            try:
                from models import Application
                db = db_module.SessionLocal()
                app_record = Application(
                    user_id=user_id,
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
            except Exception:
                # don't break prediction on DB errors; log if desired
                pass
            finally:
                try:
                    db.close()
                except Exception:
                    pass
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
            user_logged_in=bool(session.get("user_id")),
        )
    except ValueError as exc:
        flash(f"Please complete the form with valid values. {exc}", "error")
        excluded = {"ltv_ratio", "dti_ratio", "foir_ratio", "dscr"}
        filtered_specs = {k: [s for s in v if s["name"] not in excluded] 
                         for k, v in engine.feature_specs.items()}
        return (
            render_template(
                "index.html",
                datasets=DATASETS,
                selected_loan=selected_loan,
                field_specs=filtered_specs,
                current_values=current_values,
                prediction_result=None,
                metrics=engine.metrics,
                dataset_info=engine.dataset_info,
            ),
            400,
        )


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
        if selected_loan not in engine.models:
            engine.load_model(selected_loan, train_if_missing=True)
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


if __name__ == "__main__":
    app.run(debug=True)
