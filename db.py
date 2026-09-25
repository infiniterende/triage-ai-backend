from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path
from dotenv import load_dotenv
import os

from db_url import get_database_url

dotenv_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=dotenv_path)
DATABASE_URL = get_database_url()

# pool_pre_ping recovers from connections the pooler closed while idle.
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
