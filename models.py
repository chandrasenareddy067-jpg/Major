import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import relationship
from db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(50), nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)

    applications = relationship("Application", back_populates="user")


class Application(Base):
    __tablename__ = "applications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    loan_type = Column(String(80), nullable=False)
    form_data = Column(Text)
    predicted_status = Column(String(80))
    predicted_probability = Column(Float)
    ltv_ratio = Column(Float, nullable=True)
    dti_ratio = Column(Float, nullable=True)
    foir_ratio = Column(Float, nullable=True)
    dscr = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="applications")
    insight = relationship("ApplicationInsight", back_populates="application", uselist=False)


class ApplicationInsight(Base):
    __tablename__ = "application_insights"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(Integer, ForeignKey("applications.id"), nullable=False, unique=True)
    positive_signals = Column(Text)
    negative_signals = Column(Text)
    recommendations = Column(Text)
    main_positive = Column(String(255), nullable=True)
    main_negative = Column(String(255), nullable=True)
    main_recommendation = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    application = relationship("Application", back_populates="insight")
