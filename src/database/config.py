from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./medical_qa.db")

if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}, echo=False)
else:
    engine = create_engine(DATABASE_URL, echo=False)

SessionLocal = sessionmaker(bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    """Dependency that provides a database session per request.
    FastAPI calls this automatically when an endpoint needs a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
