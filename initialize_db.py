import os
import sys
from db import init_db

if __name__ == "__main__":
    # Ensure DATABASE_URL is set for PostgreSQL
    if "DATABASE_URL" not in os.environ:
        print("CRITICAL: DATABASE_URL is not set. Defaulting to local SQLite.")
    
    try:
        print("🚀 Starting database initialization...")
        init_db()
        print("✅ Database initialization complete.")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        sys.exit(1)