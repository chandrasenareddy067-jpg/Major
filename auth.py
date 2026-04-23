from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from werkzeug.security import generate_password_hash, check_password_hash
from db import SessionLocal
from models import User

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    # loan_type can be passed as query (GET) or form (POST)
    loan_type = request.args.get("loan_type") if request.method == "GET" else request.form.get("loan_type")
    if request.method == "GET":
        return render_template("auth.html", loan_type=loan_type, active_page="auth")

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    if not email or not password:
        flash("Please provide email and password.", "error")
        return redirect(url_for("auth.login", loan_type=loan_type))

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and check_password_hash(user.password_hash, password):
            session["user_id"] = user.id
            flash("Logged in successfully.", "success")
            return redirect(url_for("index", loan_type=loan_type))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for("auth.login", loan_type=loan_type))
    finally:
        db.close()


@auth_bp.route("/signup", methods=["POST"])
def signup():
    loan_type = request.form.get("loan_type")
    email = (request.form.get("email") or "").strip().lower()
    phone = (request.form.get("phone") or "").strip()
    password = request.form.get("password") or ""
    confirm = request.form.get("confirm_password") or ""

    if not email or not password:
        flash("Please provide email and password.", "error")
        return redirect(url_for("auth.login", loan_type=loan_type))
    if password != confirm:
        flash("Passwords do not match.", "error")
        return redirect(url_for("auth.login", loan_type=loan_type))

    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.email == email).first()
        if existing:
            flash("Email already registered. Please log in.", "error")
            return redirect(url_for("auth.login", loan_type=loan_type))

        user = User(email=email, phone=phone, password_hash=generate_password_hash(password))
        db.add(user)
        db.commit()
        db.refresh(user)
        session["user_id"] = user.id
        flash("Account created and logged in.", "success")
        return redirect(url_for("index", loan_type=loan_type))
    finally:
        db.close()


@auth_bp.route("/logout", methods=["GET", "POST"])
def logout():
    session.pop("user_id", None)
    flash("Logged out.", "info")
    return redirect(url_for("index"))
