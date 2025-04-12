# backend/routes/upload_route.py
from fastapi import APIRouter, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

router = APIRouter()

# Configura rutas absolutas
BASE_DIR = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

templates = Jinja2Templates(directory=TEMPLATES_DIR)
router.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

from backend.controllers.upload_controller import handle_upload

@router.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@router.post("/upload", response_class=HTMLResponse)
async def upload_file(request: Request, file: UploadFile):
    result = await handle_upload(file)
    
    return templates.TemplateResponse("upload_form.html", {
        "request": request,
        "message": result["message"],
        "result_file_path": f"/static/{result['result_file']}",
        "plot_file_path": f"/static/{result['plot_file']}"
    })

@router.get("/download-result")
async def download_result():
    result_file = os.path.join(STATIC_DIR, "results", "output.txt")
    if os.path.exists(result_file):
        return FileResponse(result_file, media_type='text/plain', filename="resultados_optimizacion.txt")
    return {"error": "El archivo de resultados no se ha generado aún."}

@router.get("/download-plot")
async def download_plot():
    plot_file = os.path.join(STATIC_DIR, "plots", "wells_plot.png")
    if os.path.exists(plot_file):
        return FileResponse(plot_file, media_type='image/png', filename="grafica_pozos.png")
    return {"error": "La gráfica no se ha generado aún."}