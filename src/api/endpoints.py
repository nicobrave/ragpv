"""
Endpoints de la API para el agente RAG.

Este módulo contiene el router de FastAPI y define todas las rutas
relacionadas con la interacción y consulta del agente.
"""
from fastapi import APIRouter, Body, HTTPException
from typing import Dict
from src.rag_engine.retriever import SupabaseRetriever
from src.rag_engine.generator import generate_response

# Crear una nueva instancia de APIRouter.
# Este objeto será usado para definir todas las rutas de este módulo.
router = APIRouter()

# Inicializamos el retriever. En una aplicación más compleja, esto se manejaría
# con un sistema de inyección de dependencias de FastAPI.
try:
    retriever = SupabaseRetriever()
except ValueError as e:
    # Si las variables de entorno no están configuradas, el retriever no se puede crear.
    retriever = None
    print(f"ADVERTENCIA: No se pudo inicializar el Retriever. {e}")

@router.post("/query", response_model=Dict[str, str])
async def handle_query(query: str = Body(..., embed=True, description="La pregunta en lenguaje natural para el agente.")):
    """
    Gestiona las consultas en lenguaje natural del usuario.

    Este endpoint orquesta el flujo RAG:
    1. Recibe la consulta del usuario.
    2. Utiliza el `SupabaseRetriever` para encontrar el contexto relevante en la base de datos.
    3. Pasa la consulta y el contexto al `Generator` para generar una respuesta.
    4. Devuelve la respuesta final.

    Args:
        query (str): La pregunta del usuario.

    Returns:
        Dict[str, str]: Un diccionario con la respuesta del agente.
    """
    if not retriever:
        raise HTTPException(
            status_code=503,
            detail="El servicio no está disponible porque el retriever no pudo ser inicializado. Verifique las variables de entorno."
        )

    # 1. Recuperar contexto de Supabase
    context = retriever.retrieve_context(query)

    # 2. Generar respuesta con el contexto
    response = generate_response(query, context)

    return {"response": response}
