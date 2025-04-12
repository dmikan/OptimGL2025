# backend/controllers/upload_controller.py
import os
from fastapi import UploadFile
from pathlib import Path
from backend.services.optimization_pipeline import run_pipeline

async def handle_upload(file: UploadFile):
    # Configura rutas absolutas
    BASE_DIR = Path(__file__).resolve().parent.parent
    UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
    RESULTS_DIR = os.path.join(BASE_DIR, "static", "results")
    PLOTS_DIR = os.path.join(BASE_DIR, "static", "plots")
    
    # Crea directorios si no existen
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Guarda el archivo subido
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Configura rutas de salida
    output_path = os.path.join(RESULTS_DIR, "output.txt")
    plot_path = os.path.join(PLOTS_DIR, "wells_plot.png")
    
    # Ejecuta el pipeline
    result_file = run_pipeline(file_path, output_file=output_path, plot_file=plot_path)

    return {
        "message": "Archivo procesado exitosamente",
        "result_file": os.path.join("results", "output.txt"),
        "plot_file": os.path.join("plots", "wells_plot.png")
    }