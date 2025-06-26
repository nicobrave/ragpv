"""
Módulo para definir los esquemas de datos Pydantic para las peticiones y respuestas de la API.

Estos esquemas son utilizados por FastAPI para la validación de datos,
serialización y documentación automática de la API.
"""
from pydantic import BaseModel
from typing import Optional, Dict, Any

class QueryRequest(BaseModel):
    """
    Esquema para una petición de consulta al agente RAG.
    """
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class QueryResponse(BaseModel):
    """
    Esquema para la respuesta de una consulta.
    """
    response: str
    source_documents: Optional[list[Dict[str, Any]]] = None
    session_id: Optional[str] = None
