from backend.services.data_loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

class Fitting():
    """Class implementing fitting of performance curves"""

    def __init__(self, q_gl, q_oil):
        # Convertir a arrays numpy y filtrar valores no válidos
        self.q_gl = np.array(q_gl, dtype=float)
        self.q_oil = np.array(q_oil, dtype=float)
        
        # Eliminar NaN o infinitos
        valid_mask = ~(np.isnan(self.q_gl) | np.isnan(self.q_oil) | 
                      np.isinf(self.q_gl) | np.isinf(self.q_oil))
        self.q_gl = self.q_gl[valid_mask]
        self.q_oil = self.q_oil[valid_mask]
        
        # Asegurar q_gl > 0 para el logaritmo
        self.q_gl = np.maximum(self.q_gl, 1e-10)

    def fit(self, model, range, maxfev=10000):
        try:
            # Valores iniciales razonables
            p0 = [100, 0.1, 100, 100, 50]  # a, b, c, d, e
            
            # Límites menos restrictivos
            bounds = (
                [-np.inf, -np.inf, 0, -1000, 0],  # Límites inferiores
                [np.inf, np.inf, 200, 1000, 150]   # Límites superiores
            )
            
            params_list, _ = curve_fit(
                model,
                self.q_gl,
                self.q_oil,
                p0=p0,
                bounds=bounds,
                maxfev=maxfev  # Aumentar iteraciones máximas
            )
            
            print("✅ Parámetros ajustados:", [f"{param:.2f}" for param in params_list])
            
            # Predecir para el rango completo
            y_pred = model(range, *params_list)
            y_pred = np.maximum(y_pred, 0)  # Asegurar producción no negativa
            
            return y_pred
            
        except Exception as e:
            print(f"❌ Error en el ajuste: {str(e)}")
            # Retornar una curva plana como fallback
            return np.zeros_like(range) + np.mean(self.q_oil)

    def model_namdar(self, q_gl_range, a, b, c, d, e):
        """Modelo mejorado con protección contra valores inválidos"""
        q_gl_range = np.maximum(q_gl_range, 1e-10)  # Evitar log(0)
        return (
            a + b * q_gl_range + c * (q_gl_range ** 0.7) +
            d * np.log(q_gl_range) + e * np.exp(-(q_gl_range ** 0.6)))
    
    def model_dan(self, q_gl_range, a, b, c, d, e):
        """Versión alternativa del modelo"""
        q_gl_range = np.maximum(q_gl_range, 1e-10)
        return (
            a + b * q_gl_range + c * (q_gl_range ** 0.5) +
            d * np.log(q_gl_range) + e * np.exp(-q_gl_range))

    def plot_fitting(self, q_gl_range, y_pred, well):
        """Visualización del ajuste"""
        plt.figure(figsize=(10, 6))
        plt.plot(q_gl_range, y_pred, label="Curva ajustada", linewidth=2)
        plt.scatter(self.q_gl, self.q_oil, color='red', label='Datos reales')
        plt.xlabel('Inyección de gas (q_gl)')
        plt.ylabel('Producción de petróleo (q_oil)')
        plt.title(f'Ajuste para Pozo {well}')
        plt.legend()
        plt.grid(True)
        plt.show()