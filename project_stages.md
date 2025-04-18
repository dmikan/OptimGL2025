#  Project Stages 
1. Use the format provided by Andres. 
2. Construct an interpolation from the format provided by Andres to one single qgl column format.
3. Modularized the code.
4. Arreglar el codigo (que em main quede más intuitivo)
5. mirar la parte del frontend (streamlit)

6. eliminar los pozos del historial con poca variación en qgl
7. actualizar el formato para conocer el nombre del pozo y el nombre de la planta e implementarlo que lo soporte el pipeline.
8. modificar la base de datos para incluir tanto el nombre del pozo como el nombre de la planta
9. la curva para producciones máximas.
10. hacer los gráficos en streamlit de cada pozo y también del general. 
11. poder tener una funcionalidad para traer la información de la base de datos de cualquier planta.
12. interfaz para ingresar datos de forma manual. 
13. verificar si es posible verificar lo del MG (con precios de qgl y oil).

#bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf], 
                                             [np.inf, np.inf, np.inf, np.inf, 110])

#bounds = ([-np.inf, -np.inf, 80, -10000, -np.inf], 
                                             [0, np.inf, np.inf, 10000, 110])                                             