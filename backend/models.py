
from sqlalchemy import create_engine, Column, Integer, String, Text, BigInteger, ForeignKey, DateTime, Enum
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import enum

Base = declarative_base()

class SourceStatus(enum.Enum):
    UPLOADED = "uploaded"
    INGESTED = "ingested"
    FAILED = "failed"

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    content = Column(Text)
    chroma_collection = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Source(Base):
    __tablename__ = 'sources'
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    lesson_id = Column(BigInteger, ForeignKey('lessons.id', ondelete='CASCADE'), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    storage_path = Column(String(1024), nullable=False)
    status = Column(Enum(SourceStatus), default=SourceStatus.UPLOADED)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    ingested_at = Column(DateTime(timezone=True), nullable=True)
