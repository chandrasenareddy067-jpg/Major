import os
from sqlalchemy import create_engine, MetaData, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///lendlogin.db")

# Render and Heroku often provide DATABASE_URL starting with 'postgres://'
# SQLAlchemy 1.4+ requires 'postgresql://'
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

SCHEMA_NAME = os.getenv("DB_SCHEMA") # Default to None for standard 'public' schema

# create engine
engine = create_engine(DATABASE_URL, future=True)

# create schema when using PostgreSQL
if SCHEMA_NAME and engine.dialect.name in ("postgresql", "postgres"):
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))
        conn.commit()

# use schema-aware metadata for databases that support it
if SCHEMA_NAME and engine.dialect.name in ("postgresql", "postgres"):
    metadata = MetaData(schema=SCHEMA_NAME)
else:
    metadata = MetaData()

Base = declarative_base(metadata=metadata)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

def init_db():
    # Import models so they are registered on Base.metadata
    try:
        import models  # noqa: F401
    except Exception:
        # If models cannot be imported, raise to surface the error
        raise
    Base.metadata.create_all(bind=engine)

    # Simple migration logic to add missing columns to the applications table
    inspector = inspect(engine)
    if "applications" in inspector.get_table_names(schema=SCHEMA_NAME):
        existing_columns = [c["name"] for c in inspector.get_columns("applications", schema=SCHEMA_NAME)]
        new_columns = {
            "ltv_ratio": "FLOAT",
            "dti_ratio": "FLOAT",
            "foir_ratio": "FLOAT",
            "dscr": "FLOAT"
        }
        
        with engine.connect() as conn:
            for col_name, col_type in new_columns.items():
                if col_name not in existing_columns:
                    table_name = f"{SCHEMA_NAME}.applications" if SCHEMA_NAME else "applications"
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_type}"))
            conn.commit()
