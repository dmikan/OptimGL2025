"""
Módulo principal de modelos.

Exporta:
- Base: Base declarativa para modelos
- Optimizacion: Modelo de optimización global
- ResultadoPozo: Modelo de resultados por pozo
- init_db: Inicializador de base de datos
- get_session: Gestor de sesiones
"""

from .database import Base, init_db, get_session
from .optimizacion import Optimizacion
from .resultado_pozo import ResultadoPozo

__all__ = [
    'Base',
    'Optimizacion',
    'ResultadoPozo',
    'init_db',
    'get_session'
]