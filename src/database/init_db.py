from src.database.config import engine, Base
from src.database.models import User, Conversation, Message

def init():
    """Create all tables in the database.
    Safe to run multiple times — it won't drop existing tables."""
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init()
