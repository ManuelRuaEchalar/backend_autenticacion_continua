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
    """
    Crea las tablas si hay base de datos configurada Y alcanzable.

    `create_engine` es perezoso: no abre conexión, así que un DATABASE_URL
    inalcanzable sólo se manifiesta aquí. Sin este try/except, una RDS caída
    o fuera de la VPC tumba el proceso entero en `run.py`, porque init_db()
    se invoca a nivel de módulo. El registro de métricas es accesorio: el
    servidor federado debe arrancar igualmente.
    """
    global engine, SessionLocal

    if not engine:
        return

    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        import logging
        logging.warning(
            f"Base de datos inalcanzable ({e.__class__.__name__}); el servidor "
            f"arranca sin registro de métricas. Detalle: {e}"
        )
        # Desactivar el engine para que get_db() devuelva None y las escrituras
        # se omitan limpiamente, en vez de reintentar y bloquear cada petición.
        engine = None
        SessionLocal = None

def get_db():
    if not SessionLocal:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
