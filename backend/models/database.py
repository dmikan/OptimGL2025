# backend/models/database.py
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

Base = declarative_base()

class Optimizacion(Base):
    __tablename__ = 'optimizaciones'
    
    id = Column(Integer, primary_key=True)
    fecha_ejecucion = Column(DateTime, default=datetime.now)
    produccion_total = Column(Float)
    qgl_total = Column(Float)
    qgl_limit = Column(Float) 
    archivo_origen = Column(String(255))
    usuario = Column(String(100))
    
    pozos = relationship("ResultadoPozo", back_populates="optimizacion")

class ResultadoPozo(Base):
    __tablename__ = 'resultados_pozos'
    
    id = Column(Integer, primary_key=True)
    optimizacion_id = Column(Integer, ForeignKey('optimizaciones.id'))
    numero_pozo = Column(Integer)
    produccion_optima = Column(Float)
    qgl_optimo = Column(Float)
    
    optimizacion = relationship("Optimizacion", back_populates="pozos")

def init_db():
    engine = create_engine('mysql+mysqlconnector://root:1030607277@localhost/optimizacion_pozos')
    Base.metadata.create_all(engine)  
    return engine

def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()