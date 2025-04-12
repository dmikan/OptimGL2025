"""
Modelos de datos y conexiones a base de datos.

Contiene:
- Base: Base declarativa para modelos SQLAlchemy
- Optimizacion: Modelo para resultados de optimización
- ResultadoPozo: Modelo para resultados por pozo
- init_db: Función para inicializar la base de datos
- get_session: Función para obtener sesión de base de datos
"""

from .database import (
    Base,
    Optimizacion,
    ResultadoPozo,
    init_db,
    get_session
)

__all__ = [
    'Base',
    'Optimizacion',
    'ResultadoPozo',
    'init_db',
    'get_session'
]