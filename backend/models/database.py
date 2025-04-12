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
    engine = create_engine('mysql+mysqlconnector://root:1030607277@localhost/optimizacion_pozos')
    Base.metadata.create_all(engine)
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()