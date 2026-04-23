from pathlib import Path
import json

from flask import Blueprint, render_template, session, redirect, url_for, flash, request, jsonify

from db import SessionLocal
from models import User, Application, ApplicationInsight
from loanlib.engine import LoanApprovalEngine
from loanlib.config import DATASETS

BASE_DIR = Path(__file__).resolve().parent

admin_bp = Blueprint("admin", __name__, url_prefix="/admin")

# Engine dedicated for admin use (loads existing models/meta from models/)
engine = LoanApprovalEngine(base_dir=BASE_DIR)


def _load_applications(db):
    apps = db.query(Application).order_by(Application.created_at.desc()).all()
    results = []
    for a in apps:
        user_email = None
        if a.user_id:
            u = db.query(User).filter(User.id == a.user_id).first()
            user_email = u.email if u else None
        try:
            form_data = json.loads(a.form_data) if a.form_data else {}
        except Exception:
            form_data = {}
        # load any stored insights for this application
        insight = db.query(ApplicationInsight).filter(ApplicationInsight.application_id == a.id).first()
        if insight:
            try:
                pos = json.loads(insight.positive_signals) if insight.positive_signals else []
            except Exception:
                pos = []
            try:
                neg = json.loads(insight.negative_signals) if insight.negative_signals else []
            except Exception:
                neg = []
            try:
                recs = json.loads(insight.recommendations) if insight.recommendations else []
            except Exception:
                recs = []
            insight_payload = {
                "positive_signals": pos,
                "negative_signals": neg,
                "recommendations": recs,
                "main_positive": insight.main_positive,
                "main_negative": insight.main_negative,
                "main_recommendation": insight.main_recommendation,
                "created_at": insight.created_at.isoformat() if insight.created_at else None,
            }
        else:
            insight_payload = None

        results.append(
            {
                "id": a.id,
                "user_id": a.user_id,
                "user_email": user_email,
                "loan_type": a.loan_type,
                "predicted_status": a.predicted_status,
                "predicted_probability": a.predicted_probability,
                "created_at": a.created_at.isoformat() if a.created_at else None,
                "form_data": form_data,
                "insight": insight_payload,
            }
        )
    return results


@admin_bp.get("/")
def dashboard():
    if not session.get("user_id"):
        flash("Please log in to view the admin dashboard.", "error")
        return redirect(url_for("auth.login"))

    db = SessionLocal()
    try:
        users_count = db.query(User).count()
        apps = _load_applications(db)
    finally:
        db.close()

    total_apps = len(apps)
    approvals = sum(1 for a in apps if (a.get("predicted_status") or "").lower() == "approved")
    rejections = total_apps - approvals

    # approval rate by loan type (aggregate from applications)
    by_loan = {}
    for a in apps:
        loan = a.get("loan_type") or "unknown"
        rec = by_loan.setdefault(loan, {"total": 0, "approved": 0})
        rec["total"] += 1
        if (a.get("predicted_status") or "").lower() == "approved":
            rec["approved"] += 1

    # Ensure every dataset key appears in loan-level stats (fill zeros when missing)
    approval_stats = {}
    for key, meta in DATASETS.items():
        v = by_loan.get(key, {"total": 0, "approved": 0})
        total = v.get("total", 0)
        approved = v.get("approved", 0)
        rate = (approved / total * 100) if total else 0
        approval_stats[key] = {"total": total, "approved": approved, "rejected": total - approved, "rate": rate, "label": meta.get("label", key)}

    # For any other loan types seen in data but not in DATASETS, include them too
    for loan_key, v in by_loan.items():
        if loan_key not in approval_stats:
            total = v.get("total", 0)
            approved = v.get("approved", 0)
            approval_stats[loan_key] = {"total": total, "approved": approved, "rejected": total - approved, "rate": (approved / total * 100 if total else 0), "label": loan_key}

    # Create an ordered list of loan stats for template iteration (preserve DATASETS order)
    loan_stats = [approval_stats[k] for k in list(DATASETS.keys()) if k in approval_stats]
    # include any extra keys at the end
    extra_keys = [k for k in approval_stats.keys() if k not in DATASETS]
    for k in extra_keys:
        loan_stats.append(approval_stats[k])

    return render_template(
        "admin.html",
        users_count=users_count,
        total_apps=total_apps,
        approvals=approvals,
        rejections=rejections,
        approval_stats=approval_stats,
        loan_stats=loan_stats,
        applications=apps,
    )


@admin_bp.get("/explain/<int:app_id>")
def explain(app_id: int):
    # return insights for a single application (computed on-demand)
    db = SessionLocal()
    try:
        app_record = db.query(Application).filter(Application.id == app_id).first()
        if not app_record:
            return jsonify({"error": "application not found"}), 404
        try:
            values = json.loads(app_record.form_data) if app_record.form_data else {}
        except Exception:
            values = {}
    finally:
        db.close()

    loan_type = app_record.loan_type
    # Ensure metadata and model are loaded for this loan type
    # Try to load a trained model without triggering training.
    has_model = False
    try:
        engine.load_model(loan_type, train_if_missing=False)
        has_model = True
    except FileNotFoundError:
        # model file missing; prepare metadata if possible and fall back to profile-based insights
        try:
            engine.prepare_metadata(loan_type)
        except Exception:
            pass

    if has_model:
        try:
            result = engine.predict(loan_type, values)
        except Exception as exc:
            return jsonify({"error": f"Could not compute explanation: {exc}"}), 500
        return jsonify({"insights": result.get("insights"), "summary": result.get("insights", {}).get("summary")})

    # Fallback: compute insights from dataset profile (no model available)
    try:
        # use stored predicted_probability if present (stored as percent), else 50%
        prob_pct = app_record.predicted_probability if app_record.predicted_probability is not None else 50.0
        probability = float(prob_pct) / 100.0
        prediction = 1 if probability >= 0.5 else 0
        insights = engine._generate_insights(loan_type, values, probability, prediction)
        return jsonify({"insights": insights, "summary": insights.get("summary")})
    except Exception as exc:
        return jsonify({"error": f"Could not compute fallback explanation: {exc}"}), 500
