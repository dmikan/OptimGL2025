from sqlalchemy import Column, Integer, Float, ForeignKey, String
from sqlalchemy.orm import relationship
from .database import Base

class ResultadoPozo(Base):
    __tablename__ = 'resultados_pozos'
    
    id = Column(Integer, primary_key=True)
    optimizacion_id = Column(Integer, ForeignKey('optimizaciones.id'))
    numero_pozo = Column(Integer)
    nombre_pozo = Column(String(100))
    produccion_optima = Column(Float)
    qgl_optimo = Column(Float)
    
    optimizacion = relationship("Optimizacion", back_populates="pozos")

