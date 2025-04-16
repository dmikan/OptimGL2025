"""
Servicios de procesamiento de datos y optimización.

Contiene:
- data_loader: Carga y preparación de datos
- fitting: Ajuste de curvas de producción
- optimization_model: Modelo matemático de optimización
- optimization_pipeline: Flujo completo de procesamiento
"""

from .data_loader import DataLoader
from .fitting import Fitting
from .optimization_model import OptimizationModel
from .optimization_pipeline import run_pipeline, fitting_group

# Intenta importar solo si existe
from .optimization_pipeline_summary import run_pipeline_summary

__all__ = [
    'DataLoader',
    'Fitting',
    'OptimizationModel',
    'run_pipeline',
    'run_pipeline_summary',
    'fitting_group'
]
