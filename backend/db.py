
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# Get the user's database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# If the URL is present, ensure it uses a synchronous driver
if DATABASE_URL and 'asyncmy' in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace('mysql+asyncmy', 'mysql+mysqlconnector')
# Fallback to old method if DATABASE_URL is not set
elif not DATABASE_URL:
    MYSQL_HOST = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")
    DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DATABASE}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
