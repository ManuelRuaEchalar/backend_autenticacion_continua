import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

class TrainingRound(Base):
    __tablename__ = 'training_rounds'
    id = Column(Integer, primary_key=True, autoincrement=True)
    round_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    n_clients = Column(Integer)
    total_samples = Column(Integer)
    metrics = Column(JSON, nullable=True)
    record_type = Column(String(50)) # 'fit' or 'evaluate'
    weighted_eer = Column(Float, nullable=True)

# Configuracion de base de datos
DATABASE_URL = os.environ.get("DATABASE_URL")

engine = None
SessionLocal = None

if DATABASE_URL:
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as e:
        import logging
        logging.error(f"Error al inicializar el engine de base de datos: {e}")

def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)

def get_db():
    if not SessionLocal:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
