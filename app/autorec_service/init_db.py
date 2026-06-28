import sqlite3
import pandas as pd
from pathlib import Path

from .config import DATABASE_PATH, PROJECT_ROOT

def initialize_database():
    """Initializes the SQLite database from CSV exports if it doesn't exist."""
    # Ensure parent directory exists
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # If the database already has tables, we assume it's initialized
    with sqlite3.connect(DATABASE_PATH) as con:
        cursor = con.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='v_doctor_features'")
        if cursor.fetchone():
            return  # Already initialized

    print("Initializing Medilink_rec database from CSV exports...")

    # Load CSVs
    exports_dir = PROJECT_ROOT / "exports"
    doctor_catalog_path = exports_dir / "doctor_catalog.csv"
    matrix_entries_path = exports_dir / "autorec_matrix_entries.csv"

    if not doctor_catalog_path.exists():
        print(f"Warning: {doctor_catalog_path} not found. Cannot initialize doctors.")
        return

    doctors_df = pd.read_csv(doctor_catalog_path)
    
    # Map columns to what v_doctor_features expects
    # doctor_cache_id,specialty_slug,specialty_name,name,headline,area,fees_egp,rating_count,waiting_time_minutes,rank_public_listing,source_url,item_text
    doctors_df["profile_text"] = doctors_df["item_text"]
    doctors_df["address_short"] = doctors_df["area"]
    
    # Calculate synthetic scores
    if "rank_public_listing" in doctors_df.columns:
        doctors_df["public_listing_score"] = (1.0 - (doctors_df["rank_public_listing"] - 1) * 0.1).clip(lower=0.0)
    else:
        doctors_df["public_listing_score"] = 0.5
        
    if "rating_count" in doctors_df.columns:
        doctors_df["rating_count_score"] = (doctors_df["rating_count"].fillna(0) / 1000.0).clip(upper=1.0)
    else:
        doctors_df["rating_count_score"] = 0.5
        
    if "waiting_time_minutes" in doctors_df.columns:
        doctors_df["waiting_time_score"] = (1.0 - (doctors_df["waiting_time_minutes"].fillna(30) / 60.0)).clip(lower=0.0)
    else:
        doctors_df["waiting_time_score"] = 0.5

    # Select columns for v_doctor_features
    columns_to_keep = [
        "doctor_cache_id", "specialty_slug", "specialty_name", "name", "headline", "area",
        "address_short", "fees_egp", "rating_count", "waiting_time_minutes", "source_url",
        "profile_text", "public_listing_score", "rating_count_score", "waiting_time_score"
    ]
    doctors_df = doctors_df[columns_to_keep]

    with sqlite3.connect(DATABASE_PATH) as con:
        # Create core tables
        con.execute("""
        CREATE TABLE IF NOT EXISTS user_profiles (
            user_id TEXT PRIMARY KEY,
            preferred_specialty_slug TEXT,
            preferred_area TEXT,
            max_fee_egp INTEGER,
            max_wait_minutes INTEGER
        )
        """)
        
        con.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_sessions (
            session_id TEXT PRIMARY KEY,
            user_id TEXT,
            channel TEXT,
            user_query TEXT,
            inferred_specialty_slug TEXT,
            requested_area TEXT,
            max_fee_egp INTEGER,
            max_wait_minutes INTEGER,
            claude_model TEXT
        )
        """)
        
        con.execute("""
        CREATE TABLE IF NOT EXISTS recommendation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_id TEXT,
            doctor_cache_id INTEGER,
            rank_position INTEGER,
            profile_match_score REAL,
            specialty_match_score REAL,
            area_match_score REAL,
            public_listing_score REAL,
            rating_count_score REAL,
            waiting_time_score REAL,
            fee_score REAL,
            interaction_score REAL,
            final_score REAL
        )
        """)
        
        con.execute("""
        CREATE TABLE IF NOT EXISTS autorec_training_runs (
            run_id TEXT PRIMARY KEY,
            include_synthetic INTEGER,
            num_users INTEGER,
            num_items INTEGER,
            num_observed INTEGER,
            hidden_dim INTEGER,
            epochs INTEGER,
            learning_rate REAL,
            train_loss REAL,
            model_path TEXT,
            metadata_path TEXT,
            notes TEXT
        )
        """)
        
        con.execute("""
        CREATE TABLE IF NOT EXISTS autorec_serving_config (
            config_key TEXT PRIMARY KEY,
            config_value TEXT
        )
        """)
        
        con.execute("""
        CREATE TABLE IF NOT EXISTS doctor_cache (
            doctor_cache_id INTEGER PRIMARY KEY
        )
        """)

        # Insert doctor IDs
        doctors_df[["doctor_cache_id"]].to_sql("doctor_cache", con, if_exists="append", index=False)
        
        # Insert v_doctor_features
        doctors_df.to_sql("v_doctor_features", con, if_exists="replace", index=False)

        # Handle matrix entries
        if matrix_entries_path.exists():
            matrix_df = pd.read_csv(matrix_entries_path)
            matrix_df.to_sql("v_autorec_matrix_entries", con, if_exists="replace", index=False)
            matrix_df.to_sql("v_autorec_matrix_entries_real", con, if_exists="replace", index=False)
        else:
            print(f"Warning: {matrix_entries_path} not found. Creating empty matrix entries table.")
            con.execute("""
            CREATE TABLE IF NOT EXISTS v_autorec_matrix_entries (
                user_id TEXT,
                doctor_cache_id INTEGER,
                interaction_score REAL
            )
            """)
            con.execute("""
            CREATE TABLE IF NOT EXISTS v_autorec_matrix_entries_real (
                user_id TEXT,
                doctor_cache_id INTEGER,
                interaction_score REAL
            )
            """)

    print("Database initialization complete.")

if __name__ == "__main__":
    initialize_database()
