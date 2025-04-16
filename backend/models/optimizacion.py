"""
Modelo de Optimización.

Contiene:
- Optimizacion: Modelo para resultados de optimización global
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime
from sqlalchemy.orm import relationship
from .database import Base

class Optimizacion(Base):
    __tablename__ = 'optimizaciones'
    
    id = Column(Integer, primary_key=True)
    fecha_ejecucion = Column(DateTime, default=datetime.now)
    produccion_total = Column(Float)
    qgl_total = Column(Float)
    qgl_limit = Column(Float)
    nombre_planta = Column(String(100))
    archivo_origen = Column(String(255))
    
    pozos = relationship("ResultadoPozo", back_populates="optimizacion")