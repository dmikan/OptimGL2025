"""
Configuración base de la base de datos.

Contiene:
- Base: Base declarativa para modelos SQLAlchemy
- init_db: Función para inicializar la base de datos
- get_session: Función para obtener sesión de base de datos
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

def init_db():
    # Configuración para MySQL
    DB_USER = "root"
    DB_PASSWORD = "1030607277"
    DB_HOST = "localhost"  # o tu dirección de servidor
    DB_NAME = "optimizacion_pozos"

    # Cadena de conexión para MySQL
    SQLALCHEMY_DATABASE_URL = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=True 
    )
    Base.metadata.create_all(engine)
    return engine

def get_session(engine=None):
    if engine is None:
        engine = init_db()
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return Session()