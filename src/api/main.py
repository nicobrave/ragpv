"""
Punto de entrada principal de la aplicación FastAPI.

Este archivo inicializa la aplicación FastAPI, configura los metadatos
y también incluye los routers de los endpoints definidos en otros módulos.
"""
from fastapi import FastAPI
from . import endpoints

# Inicializa la instancia de la aplicación FastAPI
app = FastAPI(
    title="API del Agente RAG",
    description="API para interactuar con el agente RAG para el análisis de datos operativos.",
    version="0.1.0",
    contact={
        "name": "Equipo de Desarrollo",
        "email": "dev@example.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

# Incluir el router de los endpoints de la API
# Todas las rutas definidas en `endpoints.router` tendrán el prefijo `/api`
app.include_router(endpoints.router, prefix="/api", tags=["Agente RAG"])

@app.get("/", tags=["Root"])
async def read_root():
    """
    Endpoint raíz para verificar que la API está en funcionamiento.
    
    Devuelve un mensaje de bienvenida.
    """
    return {"message": "Bienvenido a la API del Agente RAG"}
