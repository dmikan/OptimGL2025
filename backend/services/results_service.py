# backend/services/results_service.py
from backend.models.database import Optimizacion, ResultadoPozo, init_db, get_session
from datetime import datetime

def save_optimization_results(total_prod: float, 
                            total_qgl: float,
                            wells_data: list,
                            filename: str,
                            user: str = None,
                            qgl_limit: float = 1000):  # Parámetro añadido con valor por defecto
    """Guarda los resultados de optimización en la base de datos
    
    Args:
        total_prod: Producción total optimizada
        total_qgl: QGL total utilizado
        wells_data: Lista de diccionarios con datos por pozo
        filename: Nombre del archivo de entrada
        user: Usuario que ejecutó la optimización
        qgl_limit: Límite de QGL configurado para esta optimización
    """
    engine = init_db()
    session = get_session(engine)
    
    try:
        # Crear registro de optimización
        optimizacion = Optimizacion(
            produccion_total=total_prod,
            qgl_total=total_qgl,
            qgl_limit=qgl_limit,  
            archivo_origen=filename,
            usuario=user
        )
        
        session.add(optimizacion)
        session.flush()  # Para obtener el ID
        
        # Guardar resultados por pozo
        for well in wells_data:
            pozo = ResultadoPozo(
                optimizacion_id=optimizacion.id,
                numero_pozo=well['numero'],
                produccion_optima=well['produccion'],
                qgl_optimo=well['qgl']
            )
            session.add(pozo)
        
        session.commit()
        return optimizacion.id # esto es podría quitar
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()